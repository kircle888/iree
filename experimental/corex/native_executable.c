// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/corex/native_executable.h"

#include <iree/base/status.h>
#include <stddef.h>
#include <stdio.h>

#include "experimental/corex/dynamic_symbols.h"
#include "experimental/corex/pipeline_layout.h"
#include "experimental/corex/status_util.h"
#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/cuda_executable_def_reader.h"
#include "iree/schemas/cuda_executable_def_verifier.h"

typedef struct iree_hal_cuda_native_executable_t {
  iree_hal_resource_t resource;
  iree_hal_cuda_context_wrapper_t* context;
  iree_hal_pipeline_layout_t** pipeline_layouts;
  CUmodule module;
  iree_host_size_t entry_point_count;
  iree_hal_cuda_kernel_params_t entry_points[];
} iree_hal_cuda_native_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_cuda_native_executable_vtable;

static iree_hal_cuda_native_executable_t* iree_hal_cuda_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_native_executable_vtable);
  return (iree_hal_cuda_native_executable_t*)base_value;
}

iree_status_t iree_hal_cuda_native_executable_create(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_native_executable_t* executable = NULL;

  // TODO: verify the flatbuffer.
  // WARNING: THIS IS CURRENTLY INSECURE. This code is not checking any of the
  // data from the flatbuffer.
  iree_hal_cuda_ExecutableDef_table_t executable_def =
      iree_hal_cuda_ExecutableDef_as_root(
          executable_params->executable_data.data);

  flatbuffers_string_t ptx_image =
      iree_hal_cuda_ExecutableDef_ptx_image_get(executable_def);
  flatbuffers_uint32_vec_t shared_memory_sizes =
      iree_hal_cuda_ExecutableDef_shared_memory_size_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_cuda_ExecutableDef_entry_points_get(executable_def);
  iree_hal_cuda_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_cuda_ExecutableDef_block_sizes_get(executable_def);
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      total_entry_point_name_chars += flatbuffers_string_len(entry_name);
    }
  });

  // Create the kernel module.
  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_point_count * sizeof(executable->entry_points[0]) +
      total_entry_point_name_chars;
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               total_size, (void**)&executable);
  IREE_TRACE(
      char* string_table_buffer =
          (char*)((char*)executable + sizeof(*executable) +
                  entry_point_count * sizeof(executable->entry_points[0])));
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_native_executable_vtable,
                                 &executable->resource);
    executable->context = context;

    // Load the PTX image - this will fail if the device cannot handle the
    // contents. We could check this prior to creating
    int callStatus = 0;
    // if (iree_status_is_ok(status)) {
    //   callStatus = system("mkdir /tmp/iree-corex");
    //   if (callStatus) {
    //     status = iree_make_status(IREE_STATUS_UNKNOWN, "mkdir failed");
    //   }
    // }
    if (iree_status_is_ok(status)) {
      FILE* ptxFd = fopen("/tmp/iree-corex/temp1.cu", "w");
      fprintf(ptxFd, "%s", ptx_image);
      fclose(ptxFd);
    }
    if (iree_status_is_ok(status)) {
      callStatus = system(
          "/usr/local/corex-3.0.1/bin/clang++ -std=c++17 -S -emit-llvm  "
          "--cuda-device-only /tmp/iree-corex/temp1.cu -o "
          "/tmp/iree-corex/temp1.ll -O3");
      if (callStatus) {
        status = iree_make_status(IREE_STATUS_UNKNOWN, "clang++ failed");
      }
    }
    if (iree_status_is_ok(status)) {
      callStatus = system(
          "/usr/local/corex-3.0.1/bin/llc -march=bi -filetype=obj "
          "/tmp/iree-corex/temp1.ll -o "
          "/tmp/iree-corex/temp1.o");
      if (callStatus) {
        status = iree_make_status(IREE_STATUS_UNKNOWN, "lld failed");
      }
    }
    if (iree_status_is_ok(status)) {
      callStatus = system(
          "/usr/local/corex-3.0.1/bin/lld -flavor ld.lld "
          "--no-warn-missing-entry "
          "--no-undefined /tmp/iree-corex/temp1.o -o "
          "/tmp/iree-corex/temp1.cubin");
      if (callStatus) {
        status = iree_make_status(IREE_STATUS_UNKNOWN, "lld failed");
      }
    }
    if (iree_status_is_ok(status)) {
      callStatus = system(
          "/usr/local/corex-3.0.1/bin/fatbinary --cuda --64 "
          "--image=profile=ivcore10,file=/tmp/iree-corex/temp1.cubin --create "
          "/tmp/iree-corex/temp1.fatbin");
      if (callStatus) {
        status = iree_make_status(IREE_STATUS_UNKNOWN, "fatbinary failed");
      }
    }
    void* theModule;
    if (iree_status_is_ok(status)) {
      FILE* moduleFd = fopen("/tmp/iree-corex/temp1.fatbin", "rb");
      fseek(moduleFd, 0L, SEEK_END);
      size_t moduleFileSize = ftell(moduleFd);
      rewind(moduleFd);
      theModule = malloc(moduleFileSize);
      fread(theModule, 1, moduleFileSize, moduleFd);
      fclose(moduleFd);
    }
    // if (iree_status_is_ok(status)) {
    //   callStatus = system("rm -rf /tmp/iree-corex");
    //   if (callStatus) {
    //     status = iree_make_status(IREE_STATUS_UNKNOWN, "rm failed");
    //   }
    // }
    if (iree_status_is_ok(status)) {
      status = CU_RESULT_TO_STATUS(
          context->syms,
          cuModuleLoadDataEx(&executable->module, theModule, 0, NULL, NULL),
          "cuModuleLoadDataEx");
    }
  }

  if (iree_status_is_ok(status)) {
    executable->entry_point_count = entry_point_count;
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      // Lookup the function in the module; this should always succeed but we
      // cannot trust that the input was generated by our compiler.
      CUfunction function = NULL;
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      status = CU_RESULT_TO_STATUS(
          context->syms,
          cuModuleGetFunction(&function, executable->module, entry_name),
          "cuModuleGetFunction");
      if (!iree_status_is_ok(status)) break;
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "exported module function %s not found",
                                  entry_name);
        break;
      }

      int32_t max_shared_mem = 0;
      status = CU_RESULT_TO_STATUS(
          context->syms,
          cuDeviceGetAttribute(
              &max_shared_mem,
              CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
              context->cu_device),
          "cuDeviceGetAttribute");
      if (!iree_status_is_ok(status)) break;
      if (shared_memory_sizes[i] > max_shared_mem) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "CUDA driver error: Requested shared memory "
                                  "size of %d larger than allowed size of %d",
                                  shared_memory_sizes[i], max_shared_mem);
      } else {
        status = CU_RESULT_TO_STATUS(
            context->syms,
            cuFuncSetAttribute(function,
                               CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               shared_memory_sizes[i]),
            "cuFuncSetAttribute");
      }
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_cuda_kernel_params_t* params = &executable->entry_points[i];
      params->layout = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(params->layout);
      params->function = function;
      params->block_size[0] = block_sizes_vec[i].x;
      params->block_size[1] = block_sizes_vec[i].y;
      params->block_size[2] = block_sizes_vec[i].z;
      printf("%d %d %d\n",params->block_size[0],params->block_size[1],params->block_size[2]);
      params->shared_memory_size = shared_memory_sizes[i];

      // Stash the entry point name in the string table for use when tracing.
      IREE_TRACE({
        iree_host_size_t entry_name_length = flatbuffers_string_len(entry_name);
        memcpy(string_table_buffer, entry_name, entry_name_length);
        params->function_name =
            iree_make_string_view(string_table_buffer, entry_name_length);
        string_table_buffer += entry_name_length;
      });

      IREE_TRACE({
        if (iree_hal_cuda_ExecutableDef_source_locations_is_present(
                executable_def)) {
          iree_hal_cuda_FileLineLocDef_vec_t source_locs_vec =
              iree_hal_cuda_ExecutableDef_source_locations_get(executable_def);
          iree_hal_cuda_FileLineLocDef_table_t source_loc =
              iree_hal_cuda_FileLineLocDef_vec_at(source_locs_vec, i);
          flatbuffers_string_t filename =
              iree_hal_cuda_FileLineLocDef_filename_get(source_loc);
          uint32_t line = iree_hal_cuda_FileLineLocDef_line_get(source_loc);
          params->source_filename =
              iree_make_string_view(filename, flatbuffers_string_len(filename));
          params->source_line = line;
        }
      });
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_pipeline_layout_release(executable->entry_points[i].layout);
  }
  if (executable->module) {
    CUDA_IGNORE_ERROR(executable->context->syms,
                      cuModuleUnload(executable->module));
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda_native_executable_entry_point_kernel_params(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_cuda_kernel_params_t* out_params) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid entry point ordinal %d", entry_point);
  }
  memcpy(out_params, &executable->entry_points[entry_point],
         sizeof(*out_params));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_cuda_native_executable_vtable = {
        .destroy = iree_hal_cuda_native_executable_destroy,
};
