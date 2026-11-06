// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstddef>
#include <memory>
#include <new>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/polyfill_thread.h"

namespace Core {
class System;
}

namespace Tegra::Engines {
class Maxwell3D;
}

namespace VideoCommon {

/// EXPERIMENT (not for nx2): EDEN_NX2_RASTER_SPLIT=2. A third GPU stage: the split's back keeps
/// the engines' state, a "GPURaster" thread runs the rasterizer behind it. The back sends it work
/// in order (Defer); register changes reach the rasterizer's twin engines as deltas. Any other
/// thread that touches rasterizer state takes the owner lock, which the raster thread holds while
/// it runs work, so it only ever sees the rasterizer between two pieces of work.
class RasterSplit {
public:
    [[nodiscard]] static RasterSplit& Get() {
        return instance;
    }
    /// EDEN_NX2_RASTER_SPLIT=2.
    [[nodiscard]] static bool Wanted();
    [[nodiscard]] bool Active() const {
        return active.load(std::memory_order_acquire);
    }
    /// EDEN_NX2_RASTER_ASYNC=<bitmask>: which calls are queued (1 draws, 2 queries and fences,
    /// 4 uniform binds, 8 constant buffer data, 16 inline uploads, 32 the rest); default 15:
    /// with 16 something still reads inline uploads too early, and 16 or 32 hang Ocarina of Time.
    [[nodiscard]] static u32 AsyncGroups() {
        static const u32 groups = [] {
            const char* value = std::getenv("EDEN_NX2_RASTER_ASYNC");
            return value == nullptr ? 0x0fu : static_cast<u32>(std::strtoul(value, nullptr, 0));
        }();
        return groups;
    }
    [[nodiscard]] static bool OnRasterThread() {
        return on_raster_thread;
    }
    /// The thread that sends work (the split's back) while the raster stage runs.
    [[nodiscard]] bool OnProducer() const;

    void Start(Core::System& system);
    void Stop();

    /// Runs `func` on the raster thread after everything sent before it. Producer only. Small
    /// functions are built in the ring itself (no allocation).
    template <typename Func>
    void Defer(Func&& func) {
        using F = std::decay_t<Func>;
        FlushDeltas();
        Task& task = Reserve();
        if constexpr (sizeof(F) <= TaskStorage && alignof(F) <= alignof(std::max_align_t)) {
            new (task.storage) F(std::forward<Func>(func));
            task.run = [](Task& self) {
                F* const f = std::launder(reinterpret_cast<F*>(self.storage));
                (*f)();
                f->~F();
            };
        } else {
            auto* const holder = new F(std::forward<Func>(func));
            new (task.storage) F*(holder);
            task.run = [](Task& self) {
                F* const f = *std::launder(reinterpret_cast<F**>(self.storage));
                (*f)();
                delete f;
            };
        }
        Commit();
    }

    /// A register change for a twin engine, sent with the next piece of work.
    void QueueDelta(Tegra::Engines::Maxwell3D* twin, u32 method, u32 value) {
        if (delta_head - delta_flushed_tail >= DeltaRingSize) [[unlikely]] {
            WaitDeltaRoom();
        }
        delta_ring[delta_head % DeltaRingSize] = {twin, method, value};
        ++delta_head;
    }

    /// Producer: the back has executed its ring up to `position`; once the raster thread runs
    /// what was sent before this, BackDone() reaches it.
    void MarkBackPosition(u64 position) {
        if (position == marked) {
            return;
        }
        marked = position;
        Defer([this, position] { back_done.store(position, std::memory_order_release); });
    }
    /// The back ring position whose raster work is all done.
    [[nodiscard]] u64 BackDone() const {
        return back_done.load(std::memory_order_acquire);
    }
    /// Any thread: waits until the raster thread ran the work the back sent up to `position`.
    void WaitBackDone(u64 position);

    /// Producer: waits until the raster thread ran everything sent so far.
    void Drain();
    /// Any thread: waits until the raster thread ran the work sent before this call.
    void WaitSent();

    /// The owner lock: held by the raster thread while it runs work and its idle hook.
    std::mutex& OwnerLock() {
        return owner;
    }

    /// Guest memory writes still queued for the raster thread (constant buffer data).
    std::atomic<u32> pending_memory_writes{};

    /// The rasterizer's idle hook (submits deferred work), run by the raster thread when idle.
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    void RequestIdleHook();

    /// Calls nested inside a call that already holds the owner lock run inline.
    static inline thread_local u32 inline_depth{};

private:
    static constexpr size_t TaskStorage = 112;
    struct Task {
        void (*run)(Task& self);
        alignas(std::max_align_t) unsigned char storage[TaskStorage];
    };
    struct Delta {
        Tegra::Engines::Maxwell3D* twin;
        u32 method;
        u32 value;
    };
    static constexpr u64 RingSize = u64{1} << 16;
    static constexpr u64 DeltaRingSize = u64{1} << 20;

    /// The slot for the next task (waits for room); Commit makes it visible.
    Task& Reserve();
    void Commit();
    void FlushDeltas();
    void WaitDeltaRoom();
    void Loop(std::stop_token stop_token);

    static RasterSplit instance;
    static inline thread_local bool on_raster_thread{};

    Core::System* system{};
    std::atomic<bool> active{};
    std::unique_ptr<Task[]> ring;
    std::unique_ptr<Delta[]> delta_ring;
    u64 delta_head{};                       ///< Producer: next delta slot
    u64 delta_flushed{};                    ///< Producer: deltas already sent in a task
    u64 delta_flushed_tail{};               ///< Producer: oldest delta not known applied
    std::atomic<u64> delta_applied{};       ///< Raster: deltas applied
    std::atomic<u64> back_done{};           ///< Raster: back ring position fully done
    u64 marked{};                           ///< Producer: last marked back position
    std::atomic<u64> sent{};
    std::atomic<u64> done{};
    std::atomic<u32> wake{};
    std::atomic<bool> sleeping{};
    std::atomic<bool> hook_requested{};
    std::mutex owner;
    std::thread::id producer{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    std::jthread thread;
};

/// Holds the owner lock for a call from a thread other than the raster thread (after a drain
/// when called from the producer), or does nothing on the raster thread or without the stage.
class RasterOwnerScope {
public:
    explicit RasterOwnerScope(bool drain = true);
    ~RasterOwnerScope();
    RasterOwnerScope(const RasterOwnerScope&) = delete;
    RasterOwnerScope& operator=(const RasterOwnerScope&) = delete;

private:
    bool locked{};
};

} // namespace VideoCommon
