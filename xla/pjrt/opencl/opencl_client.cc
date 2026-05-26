#include "xla/pjrt/opencl/opencl_client.h"
#include "xla/pjrt/opencl/opencl_runtime.h"
#include "xla/pjrt/opencl/opencl_buffer.h"

#include <memory>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_device_description.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace minecl {

class OpenClPjRtClient;
class OpenClPjRtBuffer;

class OpenClPjRtMemorySpace final : public xla::PjRtMemorySpace {
 public:
  explicit OpenClPjRtMemorySpace(OpenClPjRtClient* client, const OpenClDeviceInfo& info)
    : client_(client),
      id_(info.id),
      debug_string_(absl::StrCat("OpenCLMemorySpace(id=", info.id, ")")),
      to_string_(debug_string_),
      c_memory_(this) {}

  PJRT_Memory* ToCApiPtr() override {
    return reinterpret_cast<PJRT_Memory*>(&c_memory_);
  }

  xla::PjRtClient* client() const override;

  absl::Span<xla::PjRtDevice* const> devices() const override {
    return absl::Span<xla::PjRtDevice* const>(
        devices_.data(), devices_.size());
  }

  int id() const override { return id_; }
  absl::string_view kind() const override { return "device"; }
  int kind_id() const override { return 0; }
  absl::string_view DebugString() const override { return debug_string_; }
  absl::string_view ToString() const override { return to_string_; }

  void AddDevice(xla::PjRtDevice* device) {
    devices_.push_back(device);
  }

 private:
  OpenClPjRtClient* client_;
  int id_;
  std::string debug_string_;
  std::string to_string_;

  std::vector<xla::PjRtDevice*> devices_;
  xla::PjRtMemorySpaceCApiDelegator c_memory_;
};

class OpenClPjRtDeviceDescription final : public xla::PjRtDeviceDescription {
 public:
  explicit OpenClPjRtDeviceDescription(const OpenClDeviceInfo& info)
      : info_(info),
        debug_string_(absl::StrCat("OpenCLDevice(id=", info.id, ")")),
        to_string_(debug_string_) {}
  int id() const override { return info_.id; }
  int process_index() const override { return 0; }

  absl::string_view device_kind() const override {
    return info_.name;
  }
  absl::string_view DebugString() const override { return debug_string_; }
  absl::string_view ToString() const override { return to_string_; }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
  Attributes() const override {
    return attributes_;
  }

 private:
  OpenClDeviceInfo info_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
  std::string debug_string_;
  std::string to_string_;
};

class OpenClPjRtDevice final : public xla::PjRtDevice {
 public:
  OpenClPjRtDevice(
    OpenClPjRtClient* client, 
    OpenClPjRtMemorySpace* memory, 
    const OpenClDeviceInfo& info,
    OpenClRuntime* runtime
  ):
  client_(client), 
  memory_(memory), 
  info_(info), 
  description_(info),
  runtime_(std::move(runtime)) {
    memory_spaces_.push_back(memory_);
  }

  const xla::PjRtDeviceDescription& description() const override {
    return description_;
  }

  xla::PjRtClient* client() const override;
  bool IsAddressable() const override { return true; }

  xla::LocalChipId local_hardware_id() const override {
    return xla::LocalChipId(info_.id);
  }

  xla::PjRtMemorySpace* memory_space() const {
    return memory_;
  }

  std::unique_ptr<xla::ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

  absl::Status TransferToInfeed(const xla::LiteralSlice& literal) override {
    return absl::UnimplementedError("OpenCL infeed not implemented");
  }

  absl::Status TransferFromOutfeed(
      xla::MutableBorrowingLiteral literal) override {
    return absl::UnimplementedError("OpenCL outfeed not implemented");
  }

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return absl::Span<xla::PjRtMemorySpace* const>(
        memory_spaces_.data(), memory_spaces_.size());
  }

  absl::StatusOr<xla::PjRtMemorySpace*> default_memory_space() const override {
    return memory_;
  }

  std::shared_ptr<OpenClRuntime> runtime() const { return runtime_; }

 private:
  OpenClPjRtClient* client_;
  OpenClPjRtMemorySpace* memory_;
  OpenClDeviceInfo info_;
  OpenClPjRtDeviceDescription description_;
  std::shared_ptr<OpenClRuntime> runtime_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
};

class OpenClPjRtClient final : public xla::PjRtClient {
 public:
  OpenClPjRtClient() {

    auto infos = EnumerateOpenClDevices();

    for (const auto& info : *infos) {

      auto runtime_or = OpenClRuntime::Create(info.platform, info.device);
      if (!runtime_or.ok()) {
        LOG(FATAL) << runtime_or.status();
      }

      std::shared_ptr<OpenClRuntime> runtime = *runtime_or;

      auto memory = std::make_unique<OpenClPjRtMemorySpace>(this, info);
      auto device = std::make_unique<OpenClPjRtDevice>(
        this, memory.get(), 
        info, runtime.get()
      );

      xla::PjRtDevice* device_ptr = device.get();
      xla::PjRtMemorySpace* memory_ptr = memory.get();

      memory->AddDevice(device_ptr);

      devices_.push_back(device_ptr);
      addressable_devices_.push_back(device_ptr);
      memory_spaces_.push_back(memory_ptr);

      owned_memory_spaces_.push_back(std::move(memory));
      owned_devices_.push_back(std::move(device));
      owned_runtimes_.push_back(std::move(runtime));
    }
  }

  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::GlobalDeviceId global_device_id) const override {
    for (xla::PjRtDevice* device : devices_) {
      if (device->id() == global_device_id.value()) {
        return device;
      }
    }
    return absl::NotFoundError("OpenCL device not found");
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::LocalDeviceId local_device_id) const override {
    for (xla::PjRtDevice* device : addressable_devices_) {
      if (device->local_hardware_id().value() == local_device_id.value()) {
        return device;
      }
    }
    return absl::NotFoundError("OpenCL addressable device not found");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
    const void* data,
    xla::PrimitiveType type,
    absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    xla::PjRtMemorySpace* memory_space,
    const xla::Layout* device_layout
  ) override {
    if (byte_strides.has_value()) {
      return absl::UnimplementedError(
          "OpenCL BufferFromHostBuffer does not support byte_strides yet");
    }

    auto* opencl_memory_space = dynamic_cast<OpenClPjRtMemorySpace*>(memory_space);

    auto devices = opencl_memory_space->devices();
    if (devices.empty()) {
      return absl::InvalidArgumentError(
          "OpenCL memory space has no attached devices");
    }

    auto* opencl_device = dynamic_cast<OpenClPjRtDevice*>(device);
    if (opencl_device == nullptr) {
      return absl::InvalidArgumentError("device is not an OpenCL device");
    }

    xla::Shape shape = xla::ShapeUtil::MakeShape(type, dims);

    const int64_t byte_size_i64 = xla::ShapeUtil::ByteSizeOf(shape);
    if (byte_size_i64 < 0) {
      return absl::InternalError("negative buffer byte size");
    }

    const size_t byte_size = static_cast<size_t>(byte_size_i64);
    TF_ASSIGN_OR_RETURN(
      cl_mem mem,
      opencl_device->runtime()->Allocate(byte_size, CL_MEM_READ_WRITE));

    auto status = opencl_device->runtime()->CopyHostToDevice(
      mem, data, byte_size
    );
    if (status != absl::OkStatus()) {
      return absl::InternalError("CopyHostToDevice failed");
    }

    auto buffer = std::make_unique<OpenClBuffer>(
      opencl_device->runtime(),
      mem, byte_size
    );

    if (on_done_with_host_buffer) {
      on_done_with_host_buffer();
    }

    std::unique_ptr<xla::PjRtBuffer> result = 
      std::make_unique<OpenClPjRtBuffer>(
        std::move(buffer),
        std::move(shape),
        opencl_device
      );

    return result;
  }

  int process_index() const override { return 0; }
  int device_count() const override { return devices_.size(); }
  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    return absl::Span<xla::PjRtDevice* const>(
        devices_.data(), devices_.size());
  }

  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return absl::Span<xla::PjRtDevice* const>(
        addressable_devices_.data(), addressable_devices_.size());
  }

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return absl::Span<xla::PjRtMemorySpace* const>(
        memory_spaces_.data(), memory_spaces_.size());
  }

  xla::PjRtPlatformId platform_id() const override {
    return xla::PjRtPlatformId(0x4F434C);  // "OCL"
  }

  absl::string_view platform_name() const override { return "opencl"; }
  absl::string_view platform_version() const override {
    return "OpenCL 3.0 / minecl experimental";
  }

 private:
  std::vector<std::unique_ptr<OpenClPjRtDevice>> owned_devices_;
  std::vector<std::shared_ptr<OpenClRuntime>> owned_runtimes_;
  std::vector<std::unique_ptr<OpenClPjRtMemorySpace>> owned_memory_spaces_;

  std::vector<xla::PjRtDevice*> devices_;
  std::vector<xla::PjRtDevice*> addressable_devices_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;

  std::shared_ptr<OpenClRuntime> runtime_;
};

class OpenClPjRtBuffer final : public xla::PjRtBuffer {
  public:
  OpenClPjRtBuffer(
    std::unique_ptr<OpenClBuffer> buffer, 
    xla::Shape shape,
    OpenClPjRtDevice* device
  ):
  buffer_(std::move(buffer)),
  shape_(std::move(shape)),
  device_(device) {}

  const xla::Shape& on_device_shape() const override {
    return shape_;
  }

  xla::PjRtDevice* device() const override {
    return device_;
  }

  xla::PjRtClient* client() const override {
    return device_->client();
  }
  
  xla::PjRtMemorySpace* memory_space() const override {
    return device_->memory_space();
  }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    if (deleted_) {
      return absl::FailedPreconditionError("buffer is deleted");
    }
    return buffer_->byte_size();
  }

  void Delete() override {
    buffer_.reset();
    deleted_ = true;
  }

  bool IsDeleted() const override {
    return deleted_;
  }

  xla::Future<> CopyRawToHost(
      void* dst,
      int64_t offset,
      int64_t transfer_size) override {
    if (deleted_) {
      return xla::Future<>(
          absl::FailedPreconditionError("buffer is deleted"));
    }

    if (offset < 0 || transfer_size < 0) {
      return xla::Future<>(
          absl::InvalidArgumentError("negative offset or transfer size"));
    }

    absl::Status status = buffer_->CopyToHost(
        dst,
        static_cast<size_t>(offset),
        static_cast<size_t>(transfer_size));

    return xla::Future<>(status);
  }

  xla::Future<> ToLiteral(xla::MutableLiteralBase* literal) override {
    if (deleted_) {
      return xla::Future<>(
          absl::FailedPreconditionError("buffer is deleted"));
    }

    return CopyRawToHost(
        literal->untyped_data(),
        0, buffer_->byte_size());
  }

  xla::Future<> GetReadyFuture() override {
    if (deleted_) {
      return xla::Future<>(
          absl::FailedPreconditionError("buffer is deleted"));
    }

    return xla::Future<>(absl::OkStatus());
  }

  bool IsOnCpu() const override {
    return false;
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
  AcquireExternalReference() override {
    return absl::UnimplementedError(
        "OpenCL AcquireExternalReference not implemented");
  }

  xla::Future<> LazyToLiteral(
      absl::AnyInvocable<xla::Future<xla::MutableLiteralBase*>() &&> generator)
      override {
    return xla::Future<>(
        absl::UnimplementedError("OpenCL LazyToLiteral not implemented"));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return absl::UnimplementedError(
        "OpenCL ReleaseDeviceMemoryOwnership not implemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyToMemorySpace(
      xla::PjRtMemorySpace* dst_memory_space) override {
    return absl::UnimplementedError(
        "OpenCL CopyToMemorySpace not implemented");
  }

  void CopyToRemoteDevice(
      xla::Future<std::string> serialized_descriptor,
      xla::PjRtBuffer::RemoteSendCallback on_done) override {
    on_done(
        absl::UnimplementedError("OpenCL CopyToRemoteDevice not implemented"),
        false);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> Bitcast(
      xla::PrimitiveType element_type,
      absl::Span<const int64_t> dims,
      const xla::Layout* device_layout) override {
    return absl::UnimplementedError("OpenCL Bitcast not implemented");
  }


  private:
  std::unique_ptr<OpenClBuffer> buffer_;
  xla::Shape shape_;
  OpenClPjRtDevice* device_;
  OpenClPjRtClient* client_;
  bool deleted_ = false;
};

xla::PjRtClient* OpenClPjRtMemorySpace::client() const {
  return reinterpret_cast<xla::PjRtClient*>(client_);
}

xla::PjRtClient* OpenClPjRtDevice::client() const {
  return reinterpret_cast<xla::PjRtClient*>(client_);
}

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetOpenClPjRtClient() {
  return std::make_unique<OpenClPjRtClient>();
}

}  // namespace minecl