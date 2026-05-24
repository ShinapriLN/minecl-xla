#include "xla/pjrt/opencl/opencl_client.h"

#include <memory>
#include <vector>
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_device_description.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace minecl {

class OpenClPjRtClient;

class OpenClPjRtMemorySpace final : public xla::PjRtMemorySpace {
 public:
  explicit OpenClPjRtMemorySpace(OpenClPjRtClient* client)
    : client_(client),
      c_memory_(this) {}

  PJRT_Memory* ToCApiPtr() override {
    return reinterpret_cast<PJRT_Memory*>(&c_memory_);
  }

  xla::PjRtClient* client() const override;

  absl::Span<xla::PjRtDevice* const> devices() const override {
    return absl::Span<xla::PjRtDevice* const>(
        devices_.data(), devices_.size());
  }

  int id() const override { return 0; }
  absl::string_view kind() const override { return "device"; }
  int kind_id() const override { return 0; }
  absl::string_view DebugString() const override { return "OpenCLMemory(id=0)"; }
  absl::string_view ToString() const override { return "OpenCLMemory(id=0)"; }

  void AddDevice(xla::PjRtDevice* device) {
    devices_.push_back(device);
  }

 private:
  OpenClPjRtClient* client_;
  std::vector<xla::PjRtDevice*> devices_;
  xla::PjRtMemorySpaceCApiDelegator c_memory_;
};

class OpenClPjRtDeviceDescription final : public xla::PjRtDeviceDescription {
 public:
  int id() const override { return 0; }
  int process_index() const override { return 0; }
  absl::string_view device_kind() const override { return "OpenCL Fake Device"; }
  absl::string_view DebugString() const override { return "OpenCLDevice(id=0)"; }
  absl::string_view ToString() const override { return "OpenCLDevice(id=0)"; }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
  Attributes() const override {
    return attributes_;
  }

 private:
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class OpenClPjRtDevice final : public xla::PjRtDevice {
  OpenClPjRtDeviceDescription description_;
 public:
  OpenClPjRtDevice(OpenClPjRtClient* client, OpenClPjRtMemorySpace* memory)
      : client_(client), memory_(memory) {

    memory_spaces_.push_back(memory_);
  }

  const xla::PjRtDeviceDescription& description() const override {
    return description_;
  }

  xla::PjRtClient* client() const override;
  bool IsAddressable() const override { return true; }
  xla::LocalChipId local_hardware_id() const override {
    return xla::LocalChipId(0);
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

 private:
  OpenClPjRtClient* client_;
  OpenClPjRtMemorySpace* memory_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
};

class OpenClPjRtClient final : public xla::PjRtClient {
 public:
  OpenClPjRtClient() {
    memory_ = std::make_unique<OpenClPjRtMemorySpace>(this);
    device_ = std::make_unique<OpenClPjRtDevice>(this, memory_.get());

    xla::PjRtDevice* device_ptr =
        static_cast<xla::PjRtDevice*>(device_.get());

    xla::PjRtMemorySpace* memory_ptr =
        static_cast<xla::PjRtMemorySpace*>(memory_.get());

    memory_->AddDevice(device_ptr);

    devices_.push_back(device_ptr);
    addressable_devices_.push_back(device_ptr);
    memory_spaces_.push_back(memory_ptr);
  }

  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::GlobalDeviceId global_device_id) const override {
    if (global_device_id.value() == 0) {
      return devices_[0];
    }
    return absl::NotFoundError("OpenCL device not found");
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::LocalDeviceId local_device_id) const override {
    if (local_device_id.value() == 0) {
      return addressable_devices_[0];
    }
    return absl::NotFoundError("OpenCL addressable device not found");
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
    return "opencl experimental";
  }

 private:
  std::unique_ptr<OpenClPjRtMemorySpace> memory_;
  std::unique_ptr<OpenClPjRtDevice> device_;

  std::vector<xla::PjRtDevice*> devices_;
  std::vector<xla::PjRtDevice*> addressable_devices_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
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