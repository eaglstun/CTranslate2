#include "ctranslate2/allocator.h"

#include "device_dispatch.h"

namespace ctranslate2 {

  Allocator& get_allocator(Device device) {
#ifdef CT2_WITH_METAL
    // Metal is intentionally kept off the DEVICE_DISPATCH path during the initial
    // bring-up: routing it through the macro would instantiate primitives<Device::METAL>
    // at every dispatch site before that specialization exists.
    if (device == Device::METAL)
      return get_allocator<Device::METAL>();
#endif
    Allocator* allocator = nullptr;
    DEVICE_DISPATCH(device, allocator = &get_allocator<D>());
    if (!allocator)
      throw std::runtime_error("No allocator defined for device " + device_to_str(device));
    return *allocator;
  }

}
