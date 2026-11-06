// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "common/logging.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/raster_split.h"

namespace VideoCommon {

namespace {
void Pause() {
#if defined(_MSC_VER) || defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}
} // Anonymous namespace

RasterSplit RasterSplit::instance;

bool RasterSplit::Wanted() {
    static const bool wanted = [] {
        const char* value = std::getenv("EDEN_NX2_RASTER_SPLIT");
        return value != nullptr && value[0] == '2';
    }();
    return wanted;
}

bool RasterSplit::OnProducer() const {
    return Active() && std::this_thread::get_id() == producer;
}

void RasterSplit::Start(Core::System& system_) {
    if (active.load(std::memory_order_relaxed)) {
        return;
    }
    system = &system_;
    ring = std::make_unique<Task[]>(RingSize);
    delta_ring = std::make_unique<Delta[]>(DeltaRingSize);
    delta_head = delta_flushed = delta_flushed_tail = 0;
    delta_applied.store(0, std::memory_order_relaxed);
    back_done.store(0, std::memory_order_relaxed);
    marked = 0;
    sent.store(0, std::memory_order_relaxed);
    done.store(0, std::memory_order_relaxed);
    pending_writes.clear();
    pending_pages.clear();
    pending_mixed = false;
    producer = std::this_thread::get_id();
    thread = std::jthread([this](std::stop_token stop_token) { Loop(stop_token); });
    active.store(true, std::memory_order_release);
    LOG_INFO(HW_GPU, "Raster split: the rasterizer runs on its own thread");
}

void RasterSplit::Stop() {
    if (!active.load(std::memory_order_relaxed)) {
        return;
    }
    Drain();
    active.store(false, std::memory_order_release);
    thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_all();
    thread = {};
}

RasterSplit::Task& RasterSplit::Reserve() {
    const u64 position = sent.load(std::memory_order_relaxed);
    for (u32 spins = 0; position - done.load(std::memory_order_acquire) >= RingSize; ++spins) {
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
    return ring[position % RingSize];
}

void RasterSplit::Commit() {
    sent.store(sent.load(std::memory_order_relaxed) + 1, std::memory_order_seq_cst);
    if (sleeping.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void RasterSplit::WaitDeltaRoom() {
    FlushDeltas();
    for (u32 spins = 0;; ++spins) {
        delta_flushed_tail = delta_applied.load(std::memory_order_acquire);
        if (delta_head - delta_flushed_tail < DeltaRingSize) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void RasterSplit::FlushDeltas() {
    if (delta_flushed == delta_head) {
        return;
    }
    const u64 begin = delta_flushed;
    const u64 end = delta_head;
    delta_flushed = end;
    Task& task = Reserve();
    struct Range {
        RasterSplit* split;
        u64 begin;
        u64 end;
    };
    new (task.storage) Range{this, begin, end};
    task.run = [](Task& self) {
        const Range range = *std::launder(reinterpret_cast<Range*>(self.storage));
        for (u64 i = range.begin; i != range.end; ++i) {
            const Delta& delta = range.split->delta_ring[i % DeltaRingSize];
            delta.twin->ApplyRasterDelta(delta.method, delta.value);
        }
        range.split->delta_applied.store(range.end, std::memory_order_release);
    };
    Commit();
}

void RasterSplit::NotePendingWrite(const void* memory_owner, u64 address, std::span<const u8> data) {
    if (data.empty()) {
        return;
    }
    PrunePendingWrites();
    if (pending_writes.empty()) {
        pending_owner = memory_owner;
        pending_mixed = false;
    } else if (pending_owner != memory_owner) {
        pending_mixed = true;
    }
    const u64 end = address + data.size();
    for (u64 page = address >> 12; page <= (end - 1) >> 12; ++page) {
        ++pending_pages[page];
    }
    pending_writes.push_back({memory_owner, address, end, sent.load(std::memory_order_relaxed),
                              std::vector<u8>(data.begin(), data.end())});
}

void RasterSplit::PrunePendingWrites() {
    const u64 completed = done.load(std::memory_order_acquire);
    while (!pending_writes.empty() && pending_writes.front().task <= completed) {
        const PendingWrite& write = pending_writes.front();
        for (u64 page = write.begin >> 12; page <= (write.end - 1) >> 12; ++page) {
            if (const auto it = pending_pages.find(page); it != pending_pages.end() && --it->second == 0) {
                pending_pages.erase(it);
            }
        }
        pending_writes.pop_front();
    }
}

bool RasterSplit::PendingWriteOverlaps(const void* memory_owner, u64 address, u64 size, bool& exact) {
    PrunePendingWrites();
    exact = !pending_mixed && (pending_writes.empty() || pending_owner == memory_owner);
    if (!exact) {
        return true;
    }
    if (size == 0) {
        return false;
    }
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        if (pending_pages.contains(page)) {
            for (const PendingWrite& write : pending_writes) {
                if (write.begin < address + size && address < write.end) {
                    return true;
                }
            }
            return false;
        }
    }
    return false;
}

void RasterSplit::OverlayPendingWrites(const void* memory_owner, u64 address, u64 size, void* out) const {
    u8* const bytes = static_cast<u8*>(out);
    for (const PendingWrite& write : pending_writes) {
        if (write.owner != memory_owner) {
            continue;
        }
        const u64 begin = (std::max)(write.begin, address);
        const u64 end = (std::min)(write.end, address + size);
        if (begin < end) {
            std::memcpy(bytes + (begin - address), write.data.data() + (begin - write.begin),
                        end - begin);
        }
    }
}

void RasterSplit::Drain() {
    if (!Active() || on_raster_thread || inline_depth != 0) {
        return;
    }
    if (std::this_thread::get_id() == producer) {
        FlushDeltas();
    }
    WaitSent();
}

void RasterSplit::WaitBackDone(u64 position) {
    if (!Active() || on_raster_thread) {
        return;
    }
    for (u32 spins = 0; back_done.load(std::memory_order_acquire) < position; ++spins) {
        if (spins < 16384) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void RasterSplit::WaitSent() {
    const u64 target = sent.load(std::memory_order_seq_cst);
    for (u32 spins = 0; done.load(std::memory_order_acquire) < target; ++spins) {
        if (spins < 16384) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void RasterSplit::RequestIdleHook() {
    hook_requested.store(true, std::memory_order_seq_cst);
    if (sleeping.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void RasterSplit::Loop(std::stop_token stop_token) {
    Common::SetCurrentThreadName("GPURaster");
    on_raster_thread = true;
    GpuSplit::MarkBackThread();
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
    u64 position = 0;
    while (!stop_token.stop_requested()) {
        u64 available = sent.load(std::memory_order_acquire);
        if (available == position) {
            for (u32 spins = 0; spins < 8192 && available == position &&
                                !hook_requested.load(std::memory_order_relaxed);
                 ++spins) {
                Pause();
                available = sent.load(std::memory_order_acquire);
            }
            if (available == position && idle_hook != nullptr) {
                hook_requested.store(false, std::memory_order_seq_cst);
                std::scoped_lock lock{owner};
                idle_hook(idle_argument);
                available = sent.load(std::memory_order_acquire);
            }
            if (available == position) {
                sleeping.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                if (sent.load(std::memory_order_seq_cst) == position &&
                    !hook_requested.load(std::memory_order_seq_cst) &&
                    !stop_token.stop_requested()) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                sleeping.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        std::scoped_lock lock{owner};
        while (position != available) {
            Task& task = ring[position % RingSize];
            task.run(task);
            ++position;
            done.store(position, std::memory_order_release);
        }
    }
}

RasterOwnerScope::RasterOwnerScope(bool drain) {
    // Only the producer used to own the rasterizer; other threads (the CPU's memory tracking)
    // called it alongside the back before and still do.
    auto& split = RasterSplit::Get();
    if (!split.OnProducer() || RasterSplit::inline_depth != 0) {
        return;
    }
    if (drain) {
        split.Drain();
    }
    split.OwnerLock().lock();
    ++RasterSplit::inline_depth;
    locked = true;
}

RasterOwnerScope::~RasterOwnerScope() {
    if (locked) {
        --RasterSplit::inline_depth;
        RasterSplit::Get().OwnerLock().unlock();
    }
}

} // namespace VideoCommon
