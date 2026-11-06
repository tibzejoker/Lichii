// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

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
    sent.store(0, std::memory_order_relaxed);
    done.store(0, std::memory_order_relaxed);
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

void RasterSplit::Push(Task task) {
    const u64 position = sent.load(std::memory_order_relaxed);
    for (u32 spins = 0; position - done.load(std::memory_order_acquire) >= RingSize; ++spins) {
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
    ring[position % RingSize] = task;
    sent.store(position + 1, std::memory_order_seq_cst);
    if (sleeping.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void RasterSplit::FlushDeltas() {
    if (deltas.empty()) {
        return;
    }
    auto* const batch = new std::vector<Delta>(std::move(deltas));
    deltas = {};
    deltas.reserve(batch->size());
    Push({[](void* argument) {
              auto* const list = static_cast<std::vector<Delta>*>(argument);
              for (const Delta& delta : *list) {
                  delta.twin->ApplyRasterDelta(delta.method, delta.value);
              }
              delete list;
          },
          batch});
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
            const Task task = ring[position % RingSize];
            task.run(task.argument);
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
