# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_hal_cts_test_suite(
#   DRIVER_NAME
#     corex
#   VARIANT_SUFFIX
#     graph
#   DRIVER_REGISTRATION_HDR
#     "runtime/src/iree/hal/drivers/corex/registration/driver_module.h"
#   DRIVER_REGISTRATION_FN
#     "iree_hal_corex_driver_module_register"
#   COMPILER_TARGET_BACKEND
#     "corex"
#   EXECUTABLE_FORMAT
#     "\"PTXE\""
#   ARGS
#     "--cuda_use_streams=false"
#   DEPS
#     iree::hal::drivers::corex::registration
#   EXCLUDED_TESTS
#     # HAL event is unimplemented for now.
#     "event"
#   LABELS
#     driver=corex
#     requires-gpu-nvidia
# )

# iree_hal_cts_test_suite(
#   DRIVER_NAME
#     corex
#   VARIANT_SUFFIX
#     stream
#   DRIVER_REGISTRATION_HDR
#     "runtime/src/iree/hal/drivers/corex/registration/driver_module.h"
#   DRIVER_REGISTRATION_FN
#     "iree_hal_corex_driver_module_register"
#   COMPILER_TARGET_BACKEND
#     "corex"
#   EXECUTABLE_FORMAT
#     "\"PTXE\""
#   ARGS
#     "--cuda_use_streams=true"
#   DEPS
#     iree::hal::drivers::corex::registration
#   EXCLUDED_TESTS
#     # HAL event is unimplemented for now.
#     "event"
#   LABELS
#     driver=corex
#     requires-gpu-nvidia
# )
