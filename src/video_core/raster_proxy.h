// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "video_core/engines/maxwell_dma.h"
#include "video_core/raster_split.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCore {

/// EXPERIMENT (not for nx2): EDEN_NX2_RASTER_SPLIT=2. What engines and the rest of the emulator
/// call instead of the rasterizer. From the split's back, calls without a result go to the raster
/// thread in order; the others drain it and run under its owner lock. Other threads take the owner
/// lock; the raster thread itself calls straight through.
class RasterProxy final : public RasterizerInterface, public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit RasterProxy(RasterizerInterface& real_) : real{real_} {}

    // Without a result: queued from the back.
    void Draw(bool is_indexed, u32 instance_count) override {
        Async(1, [r = &real, is_indexed, instance_count] { r->Draw(is_indexed, instance_count); },
              [&] { real.Draw(is_indexed, instance_count); });
    }
    void DrawIndirect() override {
        Async(1, [r = &real] { r->DrawIndirect(); }, [&] { real.DrawIndirect(); });
    }
    void DrawTexture() override {
        Async(1, [r = &real] { r->DrawTexture(); }, [&] { real.DrawTexture(); });
    }
    void Clear(u32 layer_count) override {
        Async(1, [r = &real, layer_count] { r->Clear(layer_count); }, [&] { real.Clear(layer_count); });
    }
    void ResetCounter(VideoCommon::QueryType type) override {
        Async(2, [r = &real, type] { r->ResetCounter(type); }, [&] { real.ResetCounter(type); });
    }
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override {
        Async(2, [r = &real, gpu_addr, type, flags, payload, subreport] {
                  r->Query(gpu_addr, type, flags, payload, subreport);
              },
              [&] { real.Query(gpu_addr, type, flags, payload, subreport); });
    }
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                   u32 size) override {
        Async(4, [r = &real, stage, index, gpu_addr, size] {
                  r->BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
              },
              [&] { real.BindGraphicsUniformBuffer(stage, index, gpu_addr, size); });
    }
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override {
        Async(4, [r = &real, stage, index] { r->DisableGraphicsUniformBuffer(stage, index); },
              [&] { real.DisableGraphicsUniformBuffer(stage, index); });
    }
    void SignalFence(std::function<void()>&& func) override {
        if (Queued(2)) {
            VideoCommon::RasterSplit::Get().Defer(
                [r = &real, f = std::move(func)]() mutable { r->SignalFence(std::move(f)); });
            return;
        }
        VideoCommon::RasterOwnerScope scope;
        real.SignalFence(std::move(func));
    }
    void SyncOperation(std::function<void()>&& func) override {
        if (Queued(2)) {
            VideoCommon::RasterSplit::Get().Defer(
                [r = &real, f = std::move(func)]() mutable { r->SyncOperation(std::move(f)); });
            return;
        }
        VideoCommon::RasterOwnerScope scope;
        real.SyncOperation(std::move(func));
    }
    void SignalSyncPoint(u32 value) override {
        Async(2, [r = &real, value] { r->SignalSyncPoint(value); }, [&] { real.SignalSyncPoint(value); });
    }
    void SignalReference() override {
        Async(2, [r = &real] { r->SignalReference(); }, [&] { real.SignalReference(); });
    }
    void ReleaseFences(bool force) override {
        Async(2, [r = &real, force] { r->ReleaseFences(force); }, [&] { real.ReleaseFences(force); });
    }
    void InvalidateGPUCache() override {
        Async(64, [r = &real] { r->InvalidateGPUCache(); }, [&] { real.InvalidateGPUCache(); });
    }
    void WaitForIdle() override {
        Async(128, [r = &real] { r->WaitForIdle(); }, [&] { real.WaitForIdle(); });
    }
    void FragmentBarrier() override {
        Async(256, [r = &real] { r->FragmentBarrier(); }, [&] { real.FragmentBarrier(); });
    }
    void TiledCacheBarrier() override {
        Async(512, [r = &real] { r->TiledCacheBarrier(); }, [&] { real.TiledCacheBarrier(); });
    }
    void FlushCommands() override {
        Async(1024, [r = &real] { r->FlushCommands(); }, [&] { real.FlushCommands(); });
    }
    void TickFrame() override {
        Async(2048, [r = &real] { r->TickFrame(); }, [&] { real.TickFrame(); });
    }
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override {
        if (Queued(16)) {
            auto& split = VideoCommon::RasterSplit::Get();
            split.pending_memory_writes.fetch_add(1, std::memory_order_relaxed);
            split.Defer([r = &real, address, copy_size,
                         data = std::vector<u8>(memory.begin(), memory.end())] {
                r->AccelerateInlineToMemory(address, copy_size, data);
                VideoCommon::RasterSplit::Get().pending_memory_writes.fetch_sub(
                    1, std::memory_order_relaxed);
            });
            VideoCommon::RasterSplit::last_write_queued = true;
            // EDEN_NX2_RASTER_INLINE_DRAIN=1 (bisecting): wait for it right away.
            static const bool drain_now = std::getenv("EDEN_NX2_RASTER_INLINE_DRAIN") != nullptr;
            if (drain_now) {
                split.Drain();
            }
            return;
        }
        VideoCommon::RasterOwnerScope scope;
        real.AccelerateInlineToMemory(address, copy_size, memory);
    }
    void BindChannel(Tegra::Control::ChannelState& channel) override {
        Async(4096, [r = &real, c = &channel] { r->BindChannel(*c); }, [&] { real.BindChannel(channel); });
    }
    void RegisterTransformFeedback(GPUVAddr tfb_object_addr) override {
        Async(8192, [r = &real, tfb_object_addr] { r->RegisterTransformFeedback(tfb_object_addr); },
              [&] { real.RegisterTransformFeedback(tfb_object_addr); });
    }

    // With a result, or reading engine state the raster thread doesn't mirror: synchronous.
    void DispatchCompute() override {
        Sync([&] { real.DispatchCompute(); });
    }
    void FlushAll() override {
        Sync([&] { real.FlushAll(); });
    }
    void FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override {
        Sync([&] { real.FlushRegion(addr, size, which); });
    }
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out,
                                             bool past_sync) override {
        // The split's front reads command list copies without the owner (lock-free there).
        return real.ReadGpuWrittenCommandList(addr, out, past_sync);
    }
    void NoteGpuWrittenCommandListSynced(DAddr page) override {
        Sync([&] { real.NoteGpuWrittenCommandListSynced(page); });
    }
    bool MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override {
        return Sync([&] { return real.MustFlushRegion(addr, size, which); });
    }
    RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return Sync([&] { return real.GetFlushArea(addr, size); });
    }
    bool HasPendingQueries(DAddr addr, u64 size) override {
        return Sync([&] { return real.HasPendingQueries(addr, size); });
    }
    void InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override {
        Sync([&] { real.InvalidateRegion(addr, size, which); });
    }
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override {
        Sync([&] { real.PrepareRegionWrite(addr, size, which); });
    }
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override {
        Sync([&] { real.InnerInvalidation(sequences); });
    }
    // Queued: the back calls this while it gathers CPU-dirty memory under the dirty manager's
    // lock, which a raster thread task (SignalSyncPoint -> InvalidateGPUCache) can wait for.
    void OnCacheInvalidation(PAddr addr, u64 size) override {
        Async(16384, [r = &real, addr, size] { r->OnCacheInvalidation(addr, size); },
              [&] { real.OnCacheInvalidation(addr, size); });
    }
    bool TryCacheInvalidation(DAddr addr, u64 size) override {
        return Sync([&] { return real.TryCacheInvalidation(addr, size); });
    }
    bool OnCPUWrite(PAddr addr, u64 size) override {
        return Sync([&] { return real.OnCPUWrite(addr, size); });
    }
    void UnmapMemory(DAddr addr, u64 size) override {
        Sync([&] { real.UnmapMemory(addr, size); });
    }
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override {
        Sync([&] { real.ModifyGPUMemory(as_id, addr, size); });
    }
    void FlushAndInvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) override {
        Sync([&] { real.FlushAndInvalidateRegion(addr, size, which); });
    }
    bool AccelerateConditionalRendering() override {
        return Sync([&] { return real.AccelerateConditionalRendering(); });
    }
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override {
        return Sync([&] { return real.AccelerateSurfaceCopy(src, dst, copy_config); });
    }
    bool AccelerateTemporalUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                   const Tegra::Engines::Fermi2D::Surface& dst,
                                   const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                   const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
                                   std::array<f32, 2> motion_scale, bool use_history) override {
        return Sync([&] {
            return real.AccelerateTemporalUpscale(src, dst, motion_vectors, depth, jitter,
                                                  motion_scale, use_history);
        });
    }
    bool SupportsGameDlssNetwork() const override {
        return real.SupportsGameDlssNetwork();
    }
    bool UsesNvidiaDlss() const override {
        return real.UsesNvidiaDlss();
    }
    bool CanProbeGameDlssOutput() const override {
        return real.CanProbeGameDlssOutput();
    }
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override {
        return Sync([&] { return real.ProbeGameDlssOutput(src, dst); });
    }
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override {
        return *this;
    }
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const DiskResourceLoadCallback& callback) override {
        real.LoadDiskResources(title_id, stop_loading, callback);
    }
    void InitializeChannel(Tegra::Control::ChannelState& channel) override {
        Sync([&] { real.InitializeChannel(channel); });
    }
    void ReleaseChannel(s32 channel_id) override {
        Sync([&] { real.ReleaseChannel(channel_id); });
    }
    bool HasDrawTransformFeedback() override {
        return Sync([&] { return real.HasDrawTransformFeedback(); });
    }

    // AccelerateDMAInterface: the rasterizer's, under the owner lock.
    bool BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) override {
        return Sync([&] { return real.AccessAccelerateDMA().BufferCopy(src_address, dest_address, amount); });
    }
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override {
        return Sync([&] { return real.AccessAccelerateDMA().BufferClear(src_address, amount, value); });
    }
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override {
        return Sync([&] { return real.AccessAccelerateDMA().ImageToBuffer(copy_info, src, dst); });
    }
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override {
        return Sync([&] { return real.AccessAccelerateDMA().BufferToImage(copy_info, src, dst); });
    }

    [[nodiscard]] RasterizerInterface& Real() {
        return real;
    }

private:
    /// Whether a call without a result is queued: from the producer, outside an owner scope.
    [[nodiscard]] static bool Queued(u32 group) {
        auto& split = VideoCommon::RasterSplit::Get();
        // 32 queues every call of the groups above it (64 and up: one bit per call).
        const u32 groups = VideoCommon::RasterSplit::AsyncGroups();
        const bool wanted = (groups & group) != 0 || (group >= 64 && (groups & 32) != 0);
        return wanted && split.OnProducer() &&
               VideoCommon::RasterSplit::inline_depth == 0;
    }
    template <typename Queue, typename Direct>
    void Async(u32 group, Queue&& queue, Direct&& direct) {
        if (Queued(group)) {
            VideoCommon::RasterSplit::Get().Defer(std::forward<Queue>(queue));
            return;
        }
        VideoCommon::RasterOwnerScope scope;
        direct();
    }
    template <typename Func>
    auto Sync(Func&& func) {
        VideoCommon::RasterOwnerScope scope;
        return func();
    }

    RasterizerInterface& real;
};

} // namespace VideoCore
