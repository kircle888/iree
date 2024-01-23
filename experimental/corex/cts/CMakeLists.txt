# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    corex
  DRIVER_REGISTRATION_HDR
    "experimental/corex/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_corex_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  DEPS
    iree::experimental::corex::registration
  EXCLUDED_TESTS
    # Semaphores are not fully implemented in the CUDA backend yet.
    "semaphore"
  LABELS
    driver=corex
    # requires-gpu-corex
)

# Variant test suite using graph command buffers (--cuda_use_streams=0)
iree_hal_cts_test_suite(
  DRIVER_NAME
    corex
  VARIANT_SUFFIX
    graph
  DRIVER_REGISTRATION_HDR
    "experimental/corex/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_corex_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--cuda_use_streams=0"
  DEPS
    iree::experimental::corex::registration
  INCLUDED_TESTS
    "command_buffer"
    "command_buffer_dispatch"
  LABELS
    driver=corex
    # requires-gpu-corex
)
