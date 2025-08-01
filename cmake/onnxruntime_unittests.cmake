# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
if (IOS)
  find_package(XCTest REQUIRED)
endif()

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT})
if (onnxruntime_ENABLE_TRAINING)
  list(APPEND TEST_INC_DIR ${ORTTRAINING_ROOT})
endif()

set(disabled_warnings)
function(AddTest)
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS;TEST_ARGS" ${ARGN})
  list(REMOVE_DUPLICATES _UT_SOURCES)

  if (IOS)
    onnxruntime_add_executable(${_UT_TARGET} ${TEST_SRC_DIR}/xctest/orttestmain.m)
  else()
    onnxruntime_add_executable(${_UT_TARGET} ${_UT_SOURCES})
  endif()
  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()

  source_group(TREE ${REPO_ROOT} FILES ${_UT_SOURCES})

  if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings, they are dangerous
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")
  endif()
  if (MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6330>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6330>")
    #Abseil has a lot of C4127/C4324 warnings.
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4127>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4127>")
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4324>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4324>")
  endif()

  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest")

  if (MSVC)
    # set VS debugger working directory to the test program's directory
    set_target_properties(${_UT_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>)
  endif()

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)

  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
            Threads::Threads)
    target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_ONNXRUNTIME_DLL)
  else()
    if(onnxruntime_USE_CUDA OR onnxruntime_USE_NV)
      #XXX: we should not need to do this. onnxruntime_test_all.exe should not have direct dependency on CUDA DLLs,
      # otherwise it will impact when CUDA DLLs can be unloaded.
      target_link_libraries(${_UT_TARGET} PRIVATE CUDA::cudart)
    endif()
    if(onnxruntime_USE_CUDA AND NOT onnxruntime_CUDA_MINIMAL)
      target_link_libraries(${_UT_TARGET} PRIVATE cudnn_frontend)
    endif()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()

  onnxruntime_add_include_to_target(${_UT_TARGET} date::date flatbuffers::flatbuffers)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
  if (onnxruntime_USE_CUDA)
    target_include_directories(${_UT_TARGET} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    if (onnxruntime_USE_NCCL)
      target_include_directories(${_UT_TARGET} PRIVATE ${NCCL_INCLUDE_DIRS})
    endif()
    if(onnxruntime_CUDA_MINIMAL)
      target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_CUDA_MINIMAL)
    endif()
  endif()
  if (onnxruntime_USE_TENSORRT)
    # used for instantiating placeholder TRT builder to mitigate TRT library load/unload overhead
    target_include_directories(${_UT_TARGET} PRIVATE ${TENSORRT_INCLUDE_DIR})
  endif()
  if (onnxruntime_USE_NV)
    # used for instantiating placeholder TRT builder to mitigate TRT library load/unload overhead
    target_include_directories(${_UT_TARGET} PRIVATE ${NV_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  endif()

  if(MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()

  if (WIN32)
    # include dbghelp in case tests throw an ORT exception, as that exception includes a stacktrace, which requires dbghelp.
    target_link_libraries(${_UT_TARGET} PRIVATE debug dbghelp)

    if (MSVC)
      # warning C6326: Potential comparison of a constant with another constant.
      # Lot of such things came from gtest
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      # Raw new and delete. A lot of such things came from googletest.
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      # "Global initializer calls a non-constexpr function."
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
    endif()
    target_compile_options(${_UT_TARGET} PRIVATE ${disabled_warnings})
  else()
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    if (${HAS_NOERROR})
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=uninitialized>")
    endif()
    if (${HAS_CHARACTER_CONVERSION})
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=character-conversion>")
    endif()
  endif()

  set(TEST_ARGS ${_UT_TEST_ARGS})
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # WebAssembly use a memory file system, so we do not use full path
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<TARGET_FILE_NAME:${_UT_TARGET}>.$<CONFIG>.results.xml")
    else()
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
    endif()
  endif(onnxruntime_GENERATE_TEST_REPORTS)

  if (IOS)
    # target_sources(${_UT_TARGET} PRIVATE ${TEST_SRC_DIR}/xctest/orttestmain.m)

    set(_UT_IOS_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.${_UT_TARGET})
    # replace any characters that are not valid in a bundle identifier with '-'
    string(REGEX REPLACE "[^a-zA-Z0-9\\.-]" "-" _UT_IOS_BUNDLE_GUI_IDENTIFIER ${_UT_IOS_BUNDLE_GUI_IDENTIFIER})

    set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}
      MACOSX_BUNDLE_GUI_IDENTIFIER ${_UT_IOS_BUNDLE_GUI_IDENTIFIER}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES"
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")

    xctest_add_bundle(${_UT_TARGET}_xc ${_UT_TARGET}
      ${TEST_SRC_DIR}/xctest/ortxctest.m
      ${TEST_SRC_DIR}/xctest/xcgtest.mm
      ${_UT_SOURCES})
    onnxruntime_configure_target(${_UT_TARGET}_xc)
    if(_UT_DYN)
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
              Threads::Threads)
      target_compile_definitions(${_UT_TARGET}_xc PRIVATE USE_ONNXRUNTIME_DLL)
    else()
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
    endif()
    onnxruntime_add_include_to_target(${_UT_TARGET}_xc date::date flatbuffers::flatbuffers)
    target_include_directories(${_UT_TARGET}_xc PRIVATE ${TEST_INC_DIR})
    get_target_property(${_UT_TARGET}_DEFS ${_UT_TARGET} COMPILE_DEFINITIONS)
    target_compile_definitions(${_UT_TARGET}_xc PRIVATE ${_UT_TARGET}_DEFS)

    set_target_properties(${_UT_TARGET}_xc PROPERTIES FOLDER "ONNXRuntimeXCTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}_xc
      MACOSX_BUNDLE_GUI_IDENTIFIER ${_UT_IOS_BUNDLE_GUI_IDENTIFIER}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")

    # This is a workaround for an Xcode 16 / CMake issue:
    #   error: Multiple commands produce '<build>/Debug/Debug-iphonesimulator/onnxruntime_test_all.app/PlugIns'
    #       note: CreateBuildDirectory <build>/Debug/Debug-iphonesimulator/onnxruntime_test_all.app/PlugIns
    #       note: Target 'onnxruntime_test_all' (project 'onnxruntime') has create directory command with output
    #             '<build>/Debug/Debug-iphonesimulator/onnxruntime_test_all.app/PlugIns'
    #
    # It seems related to the test target (e.g., onnxruntime_test_all_xc) LIBRARY_OUTPUT_DIRECTORY property getting set
    # to "$<TARGET_BUNDLE_CONTENT_DIR:${testee}>/PlugIns" in xctest_add_bundle():
    # https://github.com/Kitware/CMake/blob/9c4a0a9ff09735b847bbbc38caf6da7f6c7238f2/Modules/FindXCTest.cmake#L159-L168
    #
    # This is the related CMake issue: https://gitlab.kitware.com/cmake/cmake/-/issues/26301
    #
    # Unsetting LIBRARY_OUTPUT_DIRECTORY avoids the build error.
    set_property(TARGET ${_UT_TARGET}_xc PROPERTY LIBRARY_OUTPUT_DIRECTORY)

    # Don't bother calling xctest_add_test() because we don't use CTest to run tests on iOS.
    # Instead, we can call 'xcodebuild test-without-building' and specify a '-destination' referring to an iOS
    # simulator or device.
    # xctest_add_test(xctest.${_UT_TARGET} ${_UT_TARGET}_xc)
  else()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # Include the Node.js helper for finding and validating Node.js and NPM
      include(node_helper.cmake)

      if (onnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER)
        add_custom_command(TARGET ${_UT_TARGET} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/package.json $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/package-lock.json $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/karma.conf.js $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${NPM_CLI} ci
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )

        set(TEST_NPM_FLAGS)
        if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
          list(APPEND TEST_NPM_FLAGS "--wasm-threads")
        endif()
        add_test(NAME ${_UT_TARGET}
          COMMAND ${NPM_CLI} test -- ${TEST_NPM_FLAGS} --entry=${_UT_TARGET} ${TEST_ARGS}
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
      else()
        set(TEST_NODE_FLAGS)

        if (onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD)
          message(WARNING "Use system `node` to test Wasm relaxed SIMD. Please make sure to install node v21 or newer.")
          set(NODE_EXECUTABLE node)
        # prefer Node from emsdk so the version is more deterministic
        elseif (DEFINED ENV{EMSDK_NODE})
          set(NODE_EXECUTABLE $ENV{EMSDK_NODE})
        else()
          message(WARNING "EMSDK_NODE environment variable was not set. Falling back to system `node`.")
          set(NODE_EXECUTABLE node)
        endif()

        add_test(NAME ${_UT_TARGET}
          COMMAND ${NODE_EXECUTABLE} ${TEST_NODE_FLAGS} ${_UT_TARGET}.js ${TEST_ARGS}
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
      endif()
      # Set test timeout to 3 hours.
      set_tests_properties(${_UT_TARGET} PROPERTIES TIMEOUT 10800)
    else()
      add_test(NAME ${_UT_TARGET}
        COMMAND ${_UT_TARGET} ${TEST_ARGS}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
      )
      # Set test timeout to 3 hours.
      set_tests_properties(${_UT_TARGET} PROPERTIES TIMEOUT 10800)
    endif()
  endif()
endfunction(AddTest)

# general program entrypoint for C++ unit tests
set(onnxruntime_unittest_main_src "${TEST_SRC_DIR}/unittest_main/test_main.cc")

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB onnxruntime_test_common_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
)

file(GLOB onnxruntime_test_quantization_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/quantization/*.cc"
  "${TEST_SRC_DIR}/quantization/*.h"
)

file(GLOB onnxruntime_test_flatbuffers_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/flatbuffers/*.cc"
  "${TEST_SRC_DIR}/flatbuffers/*.h"
)

file(GLOB onnxruntime_test_lora_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/lora/*.cc"
  "${TEST_SRC_DIR}/lora/*.h"
)

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/ir/*.cc"
    "${TEST_SRC_DIR}/ir/*.h"
    )

  file(GLOB onnxruntime_test_optimizer_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/optimizer/*.cc"
    "${TEST_SRC_DIR}/optimizer/*.h"
    )

  set(onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/framework/*.cc"
    "${TEST_SRC_DIR}/framework/*.h"
    "${TEST_SRC_DIR}/platform/*.cc"
    )

else()  # minimal and/or reduced ops build

  set(onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/*.cc"
    )

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
    list(APPEND onnxruntime_test_framework_src_patterns
      "${TEST_SRC_DIR}/framework/ort_model_only_test.cc"
    )
  endif()

  if (NOT onnxruntime_MINIMAL_BUILD)
    file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/ir/*.cc"
      "${TEST_SRC_DIR}/ir/*.h"
      )
  endif()
endif()

if((NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
   AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_optimizer_src
       "${TEST_SRC_DIR}/optimizer/runtime_optimization/graph_runtime_optimization_test.cc")
endif()

file(GLOB onnxruntime_test_training_src
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/framework/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.cc"
  )

# TODO (baijumeswani): Remove the minimal build check here.
#                      The training api tests should be runnable even on a minimal build.
#                      This requires converting all the *.onnx files to ort format.
if (NOT onnxruntime_MINIMAL_BUILD)
  if (onnxruntime_ENABLE_TRAINING_APIS)
    file(GLOB onnxruntime_test_training_api_src
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.h"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/core/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/core/*.h"
      )
  endif()
endif()

if(WIN32)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
endif()

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  if(onnxruntime_USE_CUDA)
    list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
  endif()

  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/providers/*.h"
    "${TEST_SRC_DIR}/providers/*.cc"
    "${TEST_SRC_DIR}/opaque_api/test_opaque_api.cc"
    "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
    "${TEST_SRC_DIR}/framework/TestAllocatorManager.h"
    "${TEST_SRC_DIR}/framework/test_utils.cc"
    "${TEST_SRC_DIR}/framework/test_utils.h"
  )

  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    list(APPEND onnxruntime_test_providers_src_patterns
      "${TEST_SRC_DIR}/contrib_ops/*.h"
      "${TEST_SRC_DIR}/contrib_ops/*.cc"
      "${TEST_SRC_DIR}/contrib_ops/math/*.h"
      "${TEST_SRC_DIR}/contrib_ops/math/*.cc")
  endif()

else()
  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/framework/test_utils.cc"
    "${TEST_SRC_DIR}/framework/test_utils.h"
    # TODO: Add anything that is needed for testing a minimal build
  )
endif()

file(GLOB onnxruntime_test_providers_src CONFIGURE_DEPENDS ${onnxruntime_test_providers_src_patterns})

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_cpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cpu/*"
    )
endif()

if(onnxruntime_DISABLE_ML_OPS)
  list(FILTER onnxruntime_test_providers_cpu_src EXCLUDE REGEX ".*/ml/.*")
endif()

list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cpu_src})

if (onnxruntime_USE_CUDA AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB onnxruntime_test_providers_cuda_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_src})

  if (onnxruntime_USE_CUDA_NHWC_OPS AND CUDNN_MAJOR_VERSION GREATER 8)
    file(GLOB onnxruntime_test_providers_cuda_nhwc_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/providers/cuda/nhwc/*.cc"
    )
    list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_nhwc_src})
  endif()
endif()

if (onnxruntime_USE_CANN)
  file(GLOB_RECURSE onnxruntime_test_providers_cann_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cann/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cann_src})
endif()

# Disable training ops test for minimal build as a lot of these depend on loading an onnx model.
if (NOT onnxruntime_MINIMAL_BUILD)
  if (onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE orttraining_test_trainingops_cpu_src CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/compare_provider_test_utils.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/function_op_test_utils.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/*"
      )

    if (NOT onnxruntime_ENABLE_TRAINING)
      list(REMOVE_ITEM orttraining_test_trainingops_cpu_src
        "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/tensorboard/summary_op_test.cc"
        )
    endif()

    list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cpu_src})

    if (onnxruntime_USE_CUDA)
      file(GLOB_RECURSE orttraining_test_trainingops_cuda_src CONFIGURE_DEPENDS
        "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cuda/*"
        )
      list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cuda_src})
    endif()
  endif()
endif()

if (onnxruntime_USE_DNNL)
  file(GLOB_RECURSE onnxruntime_test_providers_dnnl_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/dnnl/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_dnnl_src})
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  file(GLOB_RECURSE onnxruntime_test_providers_nnapi_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/nnapi/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_nnapi_src})
endif()

if (onnxruntime_USE_RKNPU)
  file(GLOB_RECURSE onnxruntime_test_providers_rknpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/rknpu/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_rknpu_src})
endif()

if (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_internal_testing_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/internal_testing/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_internal_testing_src})
endif()

set (ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR "${TEST_SRC_DIR}/shared_lib")
set (ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR "${TEST_SRC_DIR}/global_thread_pools")
set (ONNXRUNTIME_CUSTOM_OP_REGISTRATION_TEST_SRC_DIR "${TEST_SRC_DIR}/custom_op_registration")
set (ONNXRUNTIME_LOGGING_APIS_TEST_SRC_DIR "${TEST_SRC_DIR}/logging_apis")
set (ONNXRUNTIME_AUTOEP_TEST_SRC_DIR "${TEST_SRC_DIR}/autoep")
set (ONNXRUNTIME_EP_GRAPH_TEST_SRC_DIR "${TEST_SRC_DIR}/ep_graph")

set (onnxruntime_shared_lib_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_allocator.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_data_copy.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_loading.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_nontensor_types.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_ort_format_models.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_run_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_session_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.cc
          )

if (NOT onnxruntime_MINIMAL_BUILD)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_builder_api.cc)
endif()

if(onnxruntime_RUN_ONNX_TESTS)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_io_types.cc)
endif()

set (onnxruntime_global_thread_pools_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_main.cc
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_inference.cc)

set (onnxruntime_webgpu_external_dawn_test_SRC
          ${TEST_SRC_DIR}/webgpu/external_dawn/main.cc)

set (onnxruntime_webgpu_delay_load_test_SRC
          ${TEST_SRC_DIR}/webgpu/delay_load/main.cc)

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  onnxruntime_test_utils
  onnxruntime_common
)

set(onnxruntime_test_ir_libs
  onnxruntime_test_utils
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_optimizer_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_framework_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  ${ONNXRUNTIME_MLAS_LIBS}
  onnxruntime_common
  )

set(onnxruntime_test_server_libs
  onnxruntime_test_utils
  onnxruntime_test_utils_for_server
)

if(WIN32)
    list(APPEND onnxruntime_test_framework_libs Advapi32)
endif()

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_CANN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cann)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_VSINPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_vsinpu)
endif()

if(onnxruntime_USE_JSEP)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_js)
endif()

if(onnxruntime_USE_WEBGPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_webgpu)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_DML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dml)
endif()

if(onnxruntime_USE_DNNL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dnnl)
endif()

if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_migraphx)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_migraphx onnxruntime_providers_shared)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml coreml_proto)
endif()

if(onnxruntime_USE_ACL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_acl)
endif()

if(onnxruntime_USE_ARMNN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_armnn)
endif()

set(ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS
    # CUDA, ROCM, TENSORRT, MIGRAPHX, DNNL, and OpenVINO are dynamically loaded at runtime.
    # QNN EP can be built as either a dynamic and static libs.
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_VSINPU}
    ${PROVIDERS_JS}
    ${PROVIDERS_WEBGPU}
    ${PROVIDERS_SNPE}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_COREML}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_AZURE}
)

if (onnxruntime_BUILD_QNN_EP_STATIC_LIB)
  list(APPEND ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS onnxruntime_providers_qnn)
endif()

set(ONNXRUNTIME_TEST_LIBS
    onnxruntime_session
    ${ONNXRUNTIME_INTEROP_TEST_LIBS}
    ${onnxruntime_libs}
    ${ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_lora
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    ${ONNXRUNTIME_MLAS_LIBS}
    onnxruntime_common
    onnxruntime_flatbuffers
)

if (onnxruntime_ENABLE_TRAINING)
  set(ONNXRUNTIME_TEST_LIBS onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_TEST_LIBS})
endif()

set(onnxruntime_test_providers_libs
    onnxruntime_test_utils
    ${ONNXRUNTIME_TEST_LIBS}
  )

if(onnxruntime_USE_TENSORRT)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/tensorrt/*)
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/tensorrt_execution_provider_utils.h")
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_tensorrt)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_tensorrt onnxruntime_providers_shared)
  list(APPEND onnxruntime_test_providers_libs ${TENSORRT_LIBRARY_INFER})
endif()

if(onnxruntime_USE_NV)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nv_tensorrt_rtx/*)
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/nv_execution_provider_utils.h")
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nv_tensorrt_rtx)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nv_tensorrt_rtx onnxruntime_providers_shared)
  list(APPEND onnxruntime_test_providers_libs ${TENSORRT_LIBRARY_INFER})
endif()


if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/migraphx/*)
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/migraphx/migraphx_execution_provider_utils.h")
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_migraphx)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_migraphx onnxruntime_providers_shared)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nnapi/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_JSEP)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/js/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_js)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_js)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_js)
endif()

if(onnxruntime_USE_WEBGPU)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/webgpu/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_webgpu)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_webgpu)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_webgpu)
endif()

# QNN EP tests require CPU EP op implementations for accuracy evaluation, so disable on minimal
# or reduced op builds.
if(onnxruntime_USE_QNN AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/qnn_node_group/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/optimizer/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_qnn)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_qnn)
  if(NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_shared)
  endif()
endif()

if(onnxruntime_USE_SNPE)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/snpe/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_snpe)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_snpe)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_snpe)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/rknpu/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/coreml/*.cc)
  if(APPLE)
    list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/coreml/*.mm)
  endif()
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_coreml coreml_proto)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml coreml_proto)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_coreml coreml_proto)
endif()

if(onnxruntime_USE_XNNPACK)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/xnnpack/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_xnnpack)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_xnnpack)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_xnnpack)
endif()

if(onnxruntime_USE_AZURE)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/azure/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_azure)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_azure)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_azure)
endif()

if (onnxruntime_USE_OPENVINO)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/openvino/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_openvino)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_openvino)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_shared)
endif()

file(GLOB onnxruntime_test_framework_src CONFIGURE_DEPENDS
  ${onnxruntime_test_framework_src_patterns}
  )

#This is a small wrapper library that shouldn't use any onnxruntime internal symbols(except onnxruntime_common).
#Because it could dynamically link to onnxruntime. Otherwise you will have two copies of onnxruntime in the same
#process and you won't know which one you are testing.
onnxruntime_add_static_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
if(MSVC)
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
else()
  target_include_directories(onnxruntime_test_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  if (HAS_CHARACTER_CONVERSION)
    target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=character-conversion>")
  endif()
endif()
if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_test_utils PRIVATE ${NCCL_INCLUDE_DIRS})
endif()
onnxruntime_add_include_to_target(onnxruntime_test_utils onnxruntime_common onnxruntime_framework onnxruntime_session GTest::gtest GTest::gmock onnx onnx_proto flatbuffers::flatbuffers nlohmann_json::nlohmann_json Boost::mp11 safeint_interface Eigen3::Eigen)
if (onnxruntime_USE_DML)
  target_add_dml(onnxruntime_test_utils)
endif()
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE
        ${ONNXRUNTIME_ROOT})
set_target_properties(onnxruntime_test_utils PROPERTIES FOLDER "ONNXRuntimeTest")
source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_test_utils_src})

if(NOT IOS)
    set(onnx_test_runner_src_dir ${TEST_SRC_DIR}/onnx)
    file(GLOB onnx_test_runner_common_srcs CONFIGURE_DEPENDS
        ${onnx_test_runner_src_dir}/*.h
        ${onnx_test_runner_src_dir}/*.cc)

    list(REMOVE_ITEM onnx_test_runner_common_srcs ${onnx_test_runner_src_dir}/main.cc)

    onnxruntime_add_static_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
    if(MSVC)
      target_compile_options(onnx_test_runner_common PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    else()
      target_include_directories(onnx_test_runner_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
    endif()
    if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      #TODO: fix the warnings, they are dangerous
      target_compile_options(onnx_test_runner_common PRIVATE "/wd4244")
    endif()
    onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_common onnxruntime_framework
            onnxruntime_test_utils onnx onnx_proto re2::re2 flatbuffers::flatbuffers Boost::mp11 safeint_interface Eigen3::Eigen)

    add_dependencies(onnx_test_runner_common onnx_test_data_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_include_directories(onnx_test_runner_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})

    set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "ONNXRuntimeTest")
    set(onnx_test_runner_common_lib onnx_test_runner_common)
endif()

set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_optimizer_src}
        ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src} ${onnxruntime_test_quantization_src}
        ${onnxruntime_test_flatbuffers_src} ${onnxruntime_test_lora_src})

if (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
  if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD AND NOT onnxruntime_DISABLE_CONTRIB_OPS)
    set(onnxruntime_test_cuda_kernels_src_patterns "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/*.cc")
  endif()

  file(GLOB onnxruntime_test_providers_cuda_ut_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/test_cases/*"
    ${onnxruntime_test_cuda_kernels_src_patterns}
  )

  # onnxruntime_providers_cuda_ut is only for unittests.
  onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_ut ${onnxruntime_test_providers_cuda_ut_src} $<TARGET_OBJECTS:onnxruntime_providers_cuda_obj>)
  config_cuda_provider_shared_module(onnxruntime_providers_cuda_ut)
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda_ut GTest::gtest GTest::gmock)
  add_dependencies(onnxruntime_providers_cuda_ut onnxruntime_test_utils onnxruntime_common)
  target_include_directories(onnxruntime_providers_cuda_ut PRIVATE ${ONNXRUNTIME_ROOT}/core/mickey)
  target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE GTest::gtest GTest::gmock ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_test_utils onnxruntime_common)
  if (MSVC)
    # Cutlass code has an issue with the following:
    # warning C4100: 'magic': unreferenced formal parameter
    target_compile_options(onnxruntime_providers_cuda_ut PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4100>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4100>")
  endif()

  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda_ut)
endif()

set(all_dependencies ${onnxruntime_test_providers_dependencies} )

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND all_tests ${onnxruntime_test_training_src})
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
    list(APPEND all_tests ${onnxruntime_test_training_api_src})
endif()


if (onnxruntime_USE_OPENVINO)
  list(APPEND all_tests ${onnxruntime_test_openvino_src})
endif()

# this is only added to onnxruntime_test_framework_libs above, but we use onnxruntime_test_providers_libs for the onnxruntime_test_all target.
# for now, add it here. better is probably to have onnxruntime_test_providers_libs use the full onnxruntime_test_framework_libs
# list given it's built on top of that library and needs all the same dependencies.
if(WIN32)
  list(APPEND onnxruntime_test_providers_libs Advapi32)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (NOT onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    list(REMOVE_ITEM all_tests
      "${TEST_SRC_DIR}/framework/execution_frame_test.cc"
      "${TEST_SRC_DIR}/framework/inference_session_test.cc"
      "${TEST_SRC_DIR}/platform/barrier_test.cc"
      "${TEST_SRC_DIR}/platform/threadpool_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/controlflow/loop_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/nn/string_normalizer_test.cc"
      "${TEST_SRC_DIR}/providers/memcpy_test.cc"
    )
  endif()
  list(REMOVE_ITEM all_tests "${TEST_SRC_DIR}/providers/cpu/reduction/reduction_ops_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/tensor/grid_sample_test.cc")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR IOS)
   # Because we do not run these model tests in our web or iOS CI build pipelines, and some test code uses C++17
   # filesystem functions that are not available in the iOS version we target.
   message("Disable model tests in onnxruntime_test_all")
   list(REMOVE_ITEM all_tests
      "${TEST_SRC_DIR}/providers/cpu/model_tests.cc"
    )
endif()

set(test_all_args)
if (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV)
  # TRT EP CI takes much longer time when updating to TRT 8.2
  # So, we only run trt ep and exclude other eps to reduce CI test time.
  #
  # The test names of model tests were using sequential number in the past.
  # This PR https://github.com/microsoft/onnxruntime/pull/10220 (Please see ExpandModelName function in model_tests.cc for more details)
  # made test name contain the "ep" and "model path" information, so we can easily filter the tests using cuda ep or other ep with *cpu_* or *xxx_*.
  list(APPEND test_all_args "--gtest_filter=-*cpu_*:*cuda_*" )
endif ()
if(NOT onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
  list(REMOVE_ITEM all_tests ${TEST_SRC_DIR}/providers/cuda/cuda_provider_test.cc)
endif()
AddTest(
  TARGET onnxruntime_test_all
  SOURCES ${all_tests} ${onnxruntime_unittest_main_src}
  LIBS
    ${onnx_test_runner_common_lib} ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
    onnx_test_data_proto
  DEPENDS ${all_dependencies}
  TEST_ARGS ${test_all_args}
)
target_include_directories(onnxruntime_test_all PRIVATE ${ONNXRUNTIME_ROOT}/core/flatbuffers/schema) # ort.fbs.h

if (MSVC)
  # The warning means the type of two integral values around a binary operator is narrow than their result.
  # If we promote the two input values first, it could be more tolerant to integer overflow.
  # However, this is test code. We are less concerned.
  target_compile_options(onnxruntime_test_all PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26451>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26451>")
  target_compile_options(onnxruntime_test_all PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")

  # Avoid this compile error in graph_transform_test.cc and qdq_transformer_test.cc:
  # fatal error C1128: number of sections exceeded object file format limit: compile with /bigobj
  set_property(SOURCE "${TEST_SRC_DIR}/optimizer/graph_transform_test.cc"
                      "${TEST_SRC_DIR}/optimizer/qdq_transformer_test.cc"
               APPEND PROPERTY COMPILE_OPTIONS "/bigobj")
else()
  target_compile_options(onnxruntime_test_all PRIVATE "-Wno-parentheses")
endif()

# TODO fix shorten-64-to-32 warnings
# there are some in builds where sizeof(size_t) != sizeof(int64_t), e.g., in 'ONNX Runtime Web CI Pipeline'
if (HAS_SHORTEN_64_TO_32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  target_compile_options(onnxruntime_test_all PRIVATE -Wno-error=shorten-64-to-32)
endif()

if (UNIX AND (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV))
    # The test_main.cc includes NvInfer.h where it has many deprecated declarations
    # simply ignore them for TensorRT EP build
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
endif()

if (MSVC AND onnxruntime_ENABLE_STATIC_ANALYSIS)
# attention_op_test.cc: Function uses '49152' bytes of stack:  exceeds /analyze:stacksize '16384'..
target_compile_options(onnxruntime_test_all PRIVATE  "/analyze:stacksize 131072")
endif()

#In AIX + gcc compiler ,crash is observed with the usage of googletest EXPECT_THROW,
#because some needed symbol is garbaged out by linker.
#So, fix is to exports the symbols from executable.
#Another way is to use -Wl,-bkeepfile for each object file where EXPECT_THROW is used like below
#target_link_options(onnxruntime_test_all PRIVATE "-Wl,-bkeepfile:CMakeFiles/onnxruntime_test_all.dir${TEST_SRC_DIR}/framework/tensor_test.cc.o")
if (CMAKE_SYSTEM_NAME MATCHES "AIX" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_target_properties(onnxruntime_test_all PROPERTIES ENABLE_EXPORTS 1)
endif()

# the default logger tests conflict with the need to have an overall default logger
# so skip in this type of
target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
if (IOS)
  target_compile_definitions(onnxruntime_test_all_xc PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
endif()
if(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE)
  target_compile_definitions(onnxruntime_test_all PUBLIC -DRUN_MODELTEST_IN_DEBUG_MODE)
endif()
if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_test_all PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()

if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  target_link_libraries(onnxruntime_test_all PRIVATE Python::Python)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_DEPENDS ${TEST_SRC_DIR}/wasm/onnxruntime_test_all_adapter.js)
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_DEPENDS ${ONNXRUNTIME_ROOT}/wasm/pre.js)
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_FLAGS "-s STACK_SIZE=5242880 -s INITIAL_MEMORY=536870912 -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4294967296 -s INCOMING_MODULE_JS_API=[preRun,locateFile,arguments,onExit,wasmMemory,buffer,instantiateWasm] --pre-js \"${TEST_SRC_DIR}/wasm/onnxruntime_test_all_adapter.js\" --pre-js \"${ONNXRUNTIME_ROOT}/wasm/pre.js\" -s \"EXPORTED_RUNTIME_METHODS=['FS']\" --preload-file ${CMAKE_CURRENT_BINARY_DIR}/testdata@/testdata -s EXIT_RUNTIME=1")
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s DEFAULT_PTHREAD_STACK_SIZE=131072 -s PROXY_TO_PTHREAD=1")
  endif()
  if (onnxruntime_USE_JSEP)
    set_target_properties(onnxruntime_test_all PROPERTIES LINK_DEPENDS ${ONNXRUNTIME_ROOT}/wasm/pre-jsep.js)
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " --pre-js \"${ONNXRUNTIME_ROOT}/wasm/pre-jsep.js\"")
  endif()

  ###
  ### if you want to investigate or debug a test failure in onnxruntime_test_all, replace the following line.
  ### those flags slow down the CI test significantly, so we don't use them by default.
  ###
  #   set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=2")
  set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s SAFE_HEAP=0 -s STACK_OVERFLOW_CHECK=1")
endif()

if (onnxruntime_ENABLE_ATEN)
  target_compile_definitions(onnxruntime_test_all PRIVATE ENABLE_ATEN)
endif()

set(test_data_target onnxruntime_test_all)

onnxruntime_add_static_library(onnx_test_data_proto ${TEST_SRC_DIR}/proto/tml.proto)
add_dependencies(onnx_test_data_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
#onnx_proto target should mark this definition as public, instead of private
target_compile_definitions(onnx_test_data_proto PRIVATE "-DONNX_API=")
onnxruntime_add_include_to_target(onnx_test_data_proto onnx_proto)
if (MSVC)
    # Cutlass code has an issue with the following:
    # warning C4100: 'magic': unreferenced formal parameter
    target_compile_options(onnx_test_data_proto PRIVATE "/wd4100")
endif()
target_include_directories(onnx_test_data_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(onnx_test_data_proto PROPERTIES FOLDER "ONNXRuntimeTest")
if(NOT DEFINED onnx_SOURCE_DIR)
  find_path(onnx_SOURCE_DIR NAMES "onnx/onnx-ml.proto3" "onnx/onnx-ml.proto" REQUIRED)
endif()
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${onnx_SOURCE_DIR} TARGET onnx_test_data_proto)

#
# onnxruntime_ir_graph test data
#
set(TEST_DATA_SRC ${TEST_SRC_DIR}/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${test_data_target}>/testdata)

set(TEST_SAMPLES_SRC ${REPO_ROOT}/samples)
set(TEST_SAMPLES_DES $<TARGET_FILE_DIR:${test_data_target}>/samples)

# Copy test data from source to destination.
add_custom_command(
  TARGET ${test_data_target} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${TEST_DATA_SRC}
  ${TEST_DATA_DES})

# Copy test samples from source to destination.
add_custom_command(
  TARGET ${test_data_target} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${TEST_SAMPLES_SRC}
  ${TEST_SAMPLES_DES})

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if (onnxruntime_USE_SNPE)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${SNPE_SO_FILES} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()

  if (onnxruntime_USE_DNNL)
    if(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND onnxruntime_DNNL_OPENCL_ROOT STREQUAL "")
      message(FATAL_ERROR "--dnnl_opencl_root required")
    elseif(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
      message(FATAL_ERROR "--dnnl_gpu_runtime required")
    elseif(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
      #file(TO_CMAKE_PATH ${onnxruntime_DNNL_OPENCL_ROOT} onnxruntime_DNNL_OPENCL_ROOT)
      #set(DNNL_OCL_INCLUDE_DIR ${onnxruntime_DNNL_OPENCL_ROOT}/include)
      #set(DNNL_GPU_CMAKE_ARGS "-DDNNL_GPU_RUNTIME=OCL " "-DOPENCLROOT=${onnxruntime_DNNL_OPENCL_ROOT}")
      target_compile_definitions(onnxruntime_test_all PUBLIC -DDNNL_GPU_RUNTIME=OCL)
    endif()
    list(APPEND onnx_test_libs dnnl)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DNNL_DLL_PATH} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()
  if(WIN32)
    set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
    onnxruntime_add_static_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
    target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
    set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
    set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})
    set(GETOPT_LIB_WIDE win_getopt_wide)
  endif()
endif()


set(onnx_test_libs
  onnxruntime_test_utils
  ${ONNXRUNTIME_TEST_LIBS}
  onnx_test_data_proto
  ${onnxruntime_EXTERNAL_LIBRARIES})

if (NOT IOS)
    onnxruntime_add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
    if(MSVC)
      target_compile_options(onnx_test_runner PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
        set_target_properties(onnx_test_runner PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
      else()
        set_target_properties(onnx_test_runner PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1")
      endif()
    endif()

    target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs} nlohmann_json::nlohmann_json)
    target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})

    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      target_link_libraries(onnx_test_runner PRIVATE Python::Python)
    endif()
    set_target_properties(onnx_test_runner PROPERTIES FOLDER "ONNXRuntimeTest")

    install(TARGETS onnx_test_runner
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            BUNDLE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if(onnxruntime_BUILD_BENCHMARKS)
    SET(BENCHMARK_DIR ${TEST_SRC_DIR}/onnx/microbenchmark)
    onnxruntime_add_executable(onnxruntime_benchmark
      ${BENCHMARK_DIR}/main.cc
      ${BENCHMARK_DIR}/modeltest.cc
      ${BENCHMARK_DIR}/pooling.cc
      ${BENCHMARK_DIR}/resize.cc
      ${BENCHMARK_DIR}/batchnorm.cc
      ${BENCHMARK_DIR}/batchnorm2.cc
      ${BENCHMARK_DIR}/tptest.cc
      ${BENCHMARK_DIR}/eigen.cc
      ${BENCHMARK_DIR}/copy.cc
      ${BENCHMARK_DIR}/gelu.cc
      ${BENCHMARK_DIR}/activation.cc
      ${BENCHMARK_DIR}/quantize.cc
      ${BENCHMARK_DIR}/reduceminmax.cc
      ${BENCHMARK_DIR}/layer_normalization.cc)
    target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_compile_definitions(onnxruntime_benchmark PRIVATE BENCHMARK_STATIC_DEFINE)
    if(WIN32)
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4141>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4141>")
      # Avoid using new and delete. But this is a benchmark program, it's ok if it has a chance to leak.
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26400>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26400>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26814>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26814>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26814>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26497>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    target_link_libraries(onnxruntime_benchmark PRIVATE onnx_test_runner_common benchmark::benchmark ${onnx_test_libs})
    add_dependencies(onnxruntime_benchmark ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")

    SET(MLAS_BENCH_DIR ${TEST_SRC_DIR}/mlas/bench)
    file(GLOB_RECURSE MLAS_BENCH_SOURCE_FILES "${MLAS_BENCH_DIR}/*.cpp" "${MLAS_BENCH_DIR}/*.h")
    onnxruntime_add_executable(onnxruntime_mlas_benchmark ${MLAS_BENCH_SOURCE_FILES} ${ONNXRUNTIME_ROOT}/core/framework/error_code.cc)
    target_include_directories(onnxruntime_mlas_benchmark PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_link_libraries(onnxruntime_mlas_benchmark PRIVATE benchmark::benchmark onnxruntime_util ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_benchmark PRIVATE BENCHMARK_STATIC_DEFINE)
    if(WIN32)
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE debug Dbghelp)
      # Avoid using new and delete. But this is a benchmark program, it's ok if it has a chance to leak.
      target_compile_options(onnxruntime_mlas_benchmark PRIVATE /wd26409)
      # "Global initializer calls a non-constexpr function." BENCHMARK_CAPTURE macro needs this.
      target_compile_options(onnxruntime_mlas_benchmark PRIVATE /wd26426)
    else()
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE  ${CMAKE_DL_LIBS})
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE cpuinfo)
    endif()
    set_target_properties(onnxruntime_mlas_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  if(WIN32)
    target_compile_options(onnx_test_runner_common PRIVATE -D_CRT_SECURE_NO_WARNINGS)
  endif()

  if (NOT onnxruntime_REDUCED_OPS_BUILD AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    add_test(NAME onnx_test_pytorch_converted
      COMMAND onnx_test_runner ${onnx_SOURCE_DIR}/onnx/backend/test/data/pytorch-converted)
    add_test(NAME onnx_test_pytorch_operator
      COMMAND onnx_test_runner ${onnx_SOURCE_DIR}/onnx/backend/test/data/pytorch-operator)
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND android_shared_libs log android)
  endif()
endif()


if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if(NOT IOS)
    #perf test runner
    set(onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
    set(onnxruntime_perf_test_src_patterns
    "${onnxruntime_perf_test_src_dir}/*.cc"
    "${onnxruntime_perf_test_src_dir}/*.h")

    if(WIN32)
      list(APPEND onnxruntime_perf_test_src_patterns
        "${onnxruntime_perf_test_src_dir}/windows/*.cc"
        "${onnxruntime_perf_test_src_dir}/windows/*.h" )
    else ()
      list(APPEND onnxruntime_perf_test_src_patterns
        "${onnxruntime_perf_test_src_dir}/posix/*.cc"
        "${onnxruntime_perf_test_src_dir}/posix/*.h" )
    endif()

    file(GLOB onnxruntime_perf_test_src CONFIGURE_DEPENDS
      ${onnxruntime_perf_test_src_patterns}
      )
    onnxruntime_add_executable(onnxruntime_perf_test ${onnxruntime_perf_test_src} ${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc)
    if(MSVC)
      target_compile_options(onnxruntime_perf_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    target_include_directories(onnxruntime_perf_test PRIVATE   ${onnx_test_runner_src_dir} ${ONNXRUNTIME_ROOT}
          ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir}
          ${CMAKE_CURRENT_BINARY_DIR})

    if (WIN32)
      target_compile_options(onnxruntime_perf_test PRIVATE ${disabled_warnings})
      if (NOT DEFINED SYS_PATH_LIB)
        set(SYS_PATH_LIB shlwapi)
      endif()
    endif()

    if (onnxruntime_BUILD_SHARED_LIB)
      #It will dynamically link to onnxruntime. So please don't add onxruntime_graph/onxruntime_framework/... here.
      #onnxruntime_common is kind of ok because it is thin, tiny and totally stateless.
      set(onnxruntime_perf_test_libs
            onnx_test_runner_common onnxruntime_test_utils onnxruntime_common
            onnxruntime onnxruntime_flatbuffers onnx_test_data_proto
            ${onnxruntime_EXTERNAL_LIBRARIES}
            ${GETOPT_LIB_WIDE} ${SYS_PATH_LIB} ${CMAKE_DL_LIBS})
      if(NOT WIN32)
        if(onnxruntime_USE_SNPE)
          list(APPEND onnxruntime_perf_test_libs onnxruntime_providers_snpe)
        endif()
      endif()
      if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        list(APPEND onnxruntime_perf_test_libs ${android_shared_libs})
      endif()
      if (CMAKE_SYSTEM_NAME MATCHES "AIX")
        list(APPEND onnxruntime_perf_test_libs onnxruntime_graph onnxruntime_session onnxruntime_providers onnxruntime_framework onnxruntime_util onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 gtest absl_failure_signal_handler absl_examine_stack absl_flags_parse  absl_flags_usage absl_flags_usage_internal)
      endif()
      target_link_libraries(onnxruntime_perf_test PRIVATE ${onnxruntime_perf_test_libs} Threads::Threads)
      if (onnxruntime_USE_CUDA OR onnxruntime_USE_NV OR onnxruntime_USE_TENSORRT)
        target_link_libraries(onnxruntime_perf_test PRIVATE CUDA::cudart)
      endif()
      if(WIN32)
        target_link_libraries(onnxruntime_perf_test PRIVATE debug dbghelp advapi32)
      endif()
    else()
      target_link_libraries(onnxruntime_perf_test PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
    endif()
    set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "ONNXRuntimeTest")

endif()


  if(onnxruntime_USE_QNN)
    #qnn ctx generator
    set(ep_weight_sharing_ctx_gen_src_dir ${TEST_SRC_DIR}/ep_weight_sharing_ctx_gen)
    set(ep_weight_sharing_ctx_gen_src_patterns
    "${ep_weight_sharing_ctx_gen_src_dir}/*.cc"
    "${ep_weight_sharing_ctx_gen_src_dir}/*.h")

    file(GLOB ep_weight_sharing_ctx_gen_src CONFIGURE_DEPENDS
      ${ep_weight_sharing_ctx_gen_src_patterns}
      )
    onnxruntime_add_executable(ep_weight_sharing_ctx_gen ${ep_weight_sharing_ctx_gen_src})
    target_include_directories(ep_weight_sharing_ctx_gen PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
    if (WIN32)
      target_compile_options(ep_weight_sharing_ctx_gen PRIVATE ${disabled_warnings})
      if (NOT DEFINED SYS_PATH_LIB)
        set(SYS_PATH_LIB shlwapi)
      endif()
    endif()

    if (onnxruntime_BUILD_SHARED_LIB)
      set(ep_weight_sharing_ctx_gen_libs onnxruntime_common onnxruntime ${onnxruntime_EXTERNAL_LIBRARIES} ${GETOPT_LIB_WIDE})
      target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE ${ep_weight_sharing_ctx_gen_libs})
      if (WIN32)
        target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE debug dbghelp advapi32)
      endif()
    else()
      target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE onnxruntime_session ${onnxruntime_test_providers_libs} ${onnxruntime_EXTERNAL_LIBRARIES} ${GETOPT_LIB_WIDE})
    endif()

    set_target_properties(ep_weight_sharing_ctx_gen PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  # shared lib
  if (onnxruntime_BUILD_SHARED_LIB)
    if(WIN32)
      onnxruntime_add_executable(onnxruntime_shared_lib_dlopen_test ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/dlopen_main.cc)
      add_dependencies(onnxruntime_shared_lib_dlopen_test ${all_dependencies} onnxruntime)
      add_test(NAME onnxruntime_shared_lib_dlopen_test COMMAND onnxruntime_shared_lib_dlopen_test WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_shared_lib_dlopen_test>)
      set_target_properties(onnxruntime_shared_lib_dlopen_test PROPERTIES FOLDER "ONNXRuntimeTest")

      if (MSVC)
        # set VS debugger working directory to the test program's directory
        set_target_properties(onnxruntime_shared_lib_dlopen_test PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_shared_lib_dlopen_test>)
      endif()
    endif()
    onnxruntime_add_static_library(onnxruntime_mocked_allocator ${TEST_SRC_DIR}/util/test_allocator.cc)
    target_include_directories(onnxruntime_mocked_allocator PUBLIC ${TEST_SRC_DIR}/util/include)
    target_link_libraries(onnxruntime_mocked_allocator PRIVATE ${GSL_TARGET})
    set_target_properties(onnxruntime_mocked_allocator PROPERTIES FOLDER "ONNXRuntimeTest")

    #################################################################
    # test inference using shared lib
    set(onnxruntime_shared_lib_test_LIBS onnxruntime_mocked_allocator onnxruntime_test_utils onnxruntime_common onnx_proto)
    if(NOT WIN32)
      if(onnxruntime_USE_SNPE)
        list(APPEND onnxruntime_shared_lib_test_LIBS onnxruntime_providers_snpe)
      endif()
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      list(APPEND onnxruntime_shared_lib_test_LIBS cpuinfo)
    endif()
    if (onnxruntime_USE_CUDA)
      list(APPEND onnxruntime_shared_lib_test_LIBS)
    endif()

    if (onnxruntime_USE_TENSORRT)
      list(APPEND onnxruntime_shared_lib_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (onnxruntime_USE_NV)
      list(APPEND onnxruntime_shared_lib_test_LIBS ${TENSORRT_LIBRARY_INFER} CUDA::cudart)
    endif()
    if (onnxruntime_USE_DML)
      list(APPEND onnxruntime_shared_lib_test_LIBS d3d12.lib)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND onnxruntime_shared_lib_test_LIBS ${android_shared_libs})
    endif()

    if (CMAKE_SYSTEM_NAME MATCHES "AIX")
      list(APPEND onnxruntime_shared_lib_test_LIBS onnxruntime_graph onnxruntime_session onnxruntime_providers onnxruntime_framework onnxruntime_util onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 onnx)
    endif()

    AddTest(DYN
            TARGET onnxruntime_shared_lib_test
            SOURCES ${onnxruntime_shared_lib_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_shared_lib_test_LIBS}
            DEPENDS ${all_dependencies}
    )

    target_include_directories(onnxruntime_shared_lib_test PRIVATE ${ONNXRUNTIME_ROOT})

    if (onnxruntime_USE_CUDA)
      target_include_directories(onnxruntime_shared_lib_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_sources(onnxruntime_shared_lib_test PRIVATE ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu)
    endif()
    if (onnxruntime_USE_NV)
      target_include_directories(onnxruntime_shared_lib_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()


    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_sources(onnxruntime_shared_lib_test PRIVATE
        "${ONNXRUNTIME_ROOT}/core/platform/android/cxa_demangle.cc"
        "${TEST_SRC_DIR}/platform/android/cxa_demangle_test.cc"
      )
      target_compile_definitions(onnxruntime_shared_lib_test PRIVATE USE_DUMMY_EXA_DEMANGLE=1)
    endif()

    if (IOS)
      add_custom_command(
        TARGET onnxruntime_shared_lib_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_SRC}
        $<TARGET_FILE_DIR:onnxruntime_shared_lib_test>/testdata)
    endif()

    if (UNIX AND (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV))
        # The test_main.cc includes NvInfer.h where it has many deprecated declarations
        # simply ignore them for TensorRT EP build
        set_property(TARGET onnxruntime_shared_lib_test APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    endif()

    # test inference using global threadpools
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Android|iOS" AND NOT onnxruntime_MINIMAL_BUILD)
      AddTest(DYN
              TARGET onnxruntime_global_thread_pools_test
              SOURCES ${onnxruntime_global_thread_pools_test_SRC}
              LIBS ${onnxruntime_shared_lib_test_LIBS}
              DEPENDS ${all_dependencies}
      )
    endif()
  endif()

  # the debug node IO functionality uses static variables, so it is best tested
  # in its own process
  if(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
    AddTest(
      TARGET onnxruntime_test_debug_node_inputs_outputs
      SOURCES
        "${TEST_SRC_DIR}/debug_node_inputs_outputs/debug_node_inputs_outputs_utils_test.cc"
        "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
        "${TEST_SRC_DIR}/framework/test_utils.cc"
        "${TEST_SRC_DIR}/providers/base_tester.h"
        "${TEST_SRC_DIR}/providers/base_tester.cc"
        "${TEST_SRC_DIR}/providers/checkers.h"
        "${TEST_SRC_DIR}/providers/checkers.cc"
        "${TEST_SRC_DIR}/providers/op_tester.h"
        "${TEST_SRC_DIR}/providers/op_tester.cc"
        "${TEST_SRC_DIR}/providers/provider_test_utils.h"
        "${TEST_SRC_DIR}/providers/tester_types.h"
        ${onnxruntime_unittest_main_src}
      LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
      DEPENDS ${all_dependencies}
    )



    target_compile_definitions(onnxruntime_test_debug_node_inputs_outputs
      PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
  endif(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)

  #some ETW tools
  if(WIN32 AND onnxruntime_ENABLE_INSTRUMENT)
    onnxruntime_add_executable(generate_perf_report_from_etl ${ONNXRUNTIME_ROOT}/tool/etw/main.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(generate_perf_report_from_etl PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(generate_perf_report_from_etl PRIVATE tdh Advapi32)

    onnxruntime_add_executable(compare_two_sessions ${ONNXRUNTIME_ROOT}/tool/etw/compare_two_sessions.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(compare_two_sessions PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(compare_two_sessions PRIVATE ${GETOPT_LIB_WIDE} tdh Advapi32)
  endif()

  if(NOT onnxruntime_target_platform STREQUAL "ARM64EC")
    file(GLOB onnxruntime_mlas_test_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/mlas/unittest/*.h"
      "${TEST_SRC_DIR}/mlas/unittest/*.cpp"
    )
    onnxruntime_add_executable(onnxruntime_mlas_test ${onnxruntime_mlas_test_src})
    if(MSVC)
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /bigobj>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/bigobj>")
    endif()
    if(IOS)
      set_target_properties(onnxruntime_mlas_test PROPERTIES
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
      )
    endif()
    target_include_directories(onnxruntime_mlas_test PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnxruntime_mlas_test PRIVATE GTest::gtest GTest::gmock ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common)
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      target_link_libraries(onnxruntime_mlas_test PRIVATE cpuinfo)
    endif()
    if(NOT WIN32)
      target_link_libraries(onnxruntime_mlas_test PRIVATE  ${CMAKE_DL_LIBS})
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_link_libraries(onnxruntime_mlas_test PRIVATE ${android_shared_libs})
    endif()
    if(WIN32)
      target_link_libraries(onnxruntime_mlas_test PRIVATE debug Dbghelp Advapi32)
    endif()
    if (onnxruntime_LINK_LIBATOMIC)
      target_link_libraries(onnxruntime_mlas_test PRIVATE atomic)
    endif()
    target_link_libraries(onnxruntime_mlas_test PRIVATE Threads::Threads)
    set_target_properties(onnxruntime_mlas_test PROPERTIES FOLDER "ONNXRuntimeTest")
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
        set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
      else()
        set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
      endif()
    endif()
endif()
  # Training API Tests
  # Disabling training_api_test_trainer. CXXOPT generates a ton of warnings because of which nuget pipeline is failing.
  # TODO(askhade): Fix the warnings.
  # This has no impact on the release as the release package and the pipeline, both do not use this.
  # This is used by devs for testing training apis.
  #if (onnxruntime_ENABLE_TRAINING_APIS)
  if (0)
    # Only files in the trainer and common folder will be compiled into test trainer.
    file(GLOB training_api_test_trainer_src
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.h"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/trainer/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/trainer/*.h"
    )
    onnxruntime_add_executable(onnxruntime_test_trainer ${training_api_test_trainer_src})

    onnxruntime_add_include_to_target(onnxruntime_test_trainer onnxruntime_session
      onnxruntime_framework onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)

    set(CXXOPTS ${cxxopts_SOURCE_DIR}/include)
    target_include_directories(onnxruntime_test_trainer PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${ONNXRUNTIME_ROOT}
      ${ORTTRAINING_ROOT}
      ${CXXOPTS}
      ${extra_includes}
      ${onnxruntime_graph_header}
      ${onnxruntime_exec_src_dir}
    )

    set(ONNXRUNTIME_TEST_LIBS
      onnxruntime_session
      ${onnxruntime_libs}
      # CUDA is dynamically loaded at runtime
      onnxruntime_optimizer
      onnxruntime_providers
      onnxruntime_util
      onnxruntime_lora
      onnxruntime_framework
      onnxruntime_util
      onnxruntime_graph
      ${ONNXRUNTIME_MLAS_LIBS}
      onnxruntime_common
      onnxruntime_flatbuffers
    )

    target_link_libraries(onnxruntime_test_trainer PRIVATE
      ${ONNXRUNTIME_TEST_LIBS}
      ${onnxruntime_EXTERNAL_LIBRARIES}
    )
    set_target_properties(onnxruntime_test_trainer PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")

  set(custom_op_src_patterns
    "${TEST_SRC_DIR}/testdata/custom_op_library/*.h"
    "${TEST_SRC_DIR}/testdata/custom_op_library/*.cc"
    "${TEST_SRC_DIR}/testdata/custom_op_library/cpu/cpu_ops.*"
  )

  set(custom_op_lib_include ${REPO_ROOT}/include)
  set(custom_op_lib_option)
  set(custom_op_lib_link ${GSL_TARGET})

  if (onnxruntime_USE_CUDA)
    list(APPEND custom_op_src_patterns
        "${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu"
        "${TEST_SRC_DIR}/testdata/custom_op_library/cuda/cuda_ops.*")
    list(APPEND custom_op_lib_include ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    if (HAS_QSPECTRE)
      list(APPEND custom_op_lib_option "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /Qspectre>")
    endif()
  endif()

  file(GLOB custom_op_src ${custom_op_src_patterns})
  onnxruntime_add_shared_library(custom_op_library ${custom_op_src})
  target_compile_options(custom_op_library PRIVATE ${custom_op_lib_option})
  target_include_directories(custom_op_library PRIVATE ${REPO_ROOT}/include ${custom_op_lib_include})
  target_link_libraries(custom_op_library PRIVATE ${GSL_TARGET} ${custom_op_lib_link})

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.def")
    if (NOT onnxruntime_USE_CUDA)
      target_compile_options(custom_op_library PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
    endif()
  endif()
  set_property(TARGET custom_op_library APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})
  #ON AIX, to call dlopen on custom_op_library, we need to generate this library as shared object .so file.
  #Latest cmake behavior is changed and  cmake will remove shared object .so file after generating the shared archive.
  #To prevent that, making AIX_SHARED_LIBRARY_ARCHIVE as OFF for custom_op_library.
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    set_target_properties(custom_op_library PROPERTIES AIX_SHARED_LIBRARY_ARCHIVE OFF)
  endif()

  if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    if (onnxruntime_BUILD_JAVA AND NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
      block()
        message(STATUS "Enabling Java tests")

        # native-test is added to resources so custom_op_lib can be loaded
        # and we want to copy it there
        set(JAVA_NATIVE_TEST_DIR ${JAVA_OUTPUT_DIR}/native-test)
        file(MAKE_DIRECTORY ${JAVA_NATIVE_TEST_DIR})

        set(CUSTOM_OP_LIBRARY_DST_FILE_NAME
            $<IF:$<BOOL:${WIN32}>,$<TARGET_FILE_NAME:custom_op_library>,$<TARGET_LINKER_FILE_NAME:custom_op_library>>)

        add_custom_command(TARGET custom_op_library POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:custom_op_library>
                ${JAVA_NATIVE_TEST_DIR}/${CUSTOM_OP_LIBRARY_DST_FILE_NAME})

        # also copy other library dependencies that may be required by tests to native-test
        if(onnxruntime_USE_QNN)
          add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${QNN_LIB_FILES} ${JAVA_NATIVE_TEST_DIR})
        endif()

        # delegate to gradle's test runner

        # On Windows, ctest requires a test to be an .exe(.com) file. With gradle wrapper, we get gradlew.bat.
        # To work around this, we delegate gradle execution to a separate .cmake file that can be run with cmake.
        # For simplicity, we use this setup for all supported platforms and not just Windows.

        # Note: Here we rely on the values in ORT_PROVIDER_FLAGS to be of the format "-Doption=value".
        # This happens to also match the gradle command line option for specifying system properties.
        set(GRADLE_SYSTEM_PROPERTY_DEFINITIONS ${ORT_PROVIDER_FLAGS})

        if(onnxruntime_ENABLE_TRAINING_APIS)
          message(STATUS "Enabling Java tests for training APIs")

          list(APPEND GRADLE_SYSTEM_PROPERTY_DEFINITIONS "-DENABLE_TRAINING_APIS=1")
        endif()

        add_test(NAME onnxruntime4j_test COMMAND
            ${CMAKE_COMMAND}
                -DGRADLE_EXECUTABLE=${GRADLE_EXECUTABLE}
                -DBIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
                -DREPO_ROOT=${REPO_ROOT}
                # Note: Quotes are important here to pass a list of values as a single property.
                "-DGRADLE_SYSTEM_PROPERTY_DEFINITIONS=${GRADLE_SYSTEM_PROPERTY_DEFINITIONS}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_java_unittests.cmake)

        set_property(TEST onnxruntime4j_test APPEND PROPERTY DEPENDS onnxruntime4j_jni)
      endblock()
    endif()
  endif()

  if (onnxruntime_BUILD_SHARED_LIB AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
    set (onnxruntime_customopregistration_test_SRC
            ${ONNXRUNTIME_CUSTOM_OP_REGISTRATION_TEST_SRC_DIR}/test_registercustomops.cc)

    set(onnxruntime_customopregistration_test_LIBS custom_op_library onnxruntime_common onnxruntime_test_utils)

    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      list(APPEND onnxruntime_customopregistration_test_LIBS cpuinfo)
    endif()
    if (onnxruntime_USE_TENSORRT)
      list(APPEND onnxruntime_customopregistration_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (onnxruntime_USE_NV)
      list(APPEND onnxruntime_customopregistration_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (CMAKE_SYSTEM_NAME MATCHES "AIX")
      list(APPEND onnxruntime_customopregistration_test_LIBS onnxruntime_graph onnxruntime_session onnxruntime_providers onnxruntime_lora onnxruntime_framework onnxruntime_util onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 ${PROTOBUF_LIB} onnx onnx_proto)
    endif()
    AddTest(DYN
            TARGET onnxruntime_customopregistration_test
            SOURCES ${onnxruntime_customopregistration_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_customopregistration_test_LIBS}
            DEPENDS ${all_dependencies}
    )

    if (IOS)
      add_custom_command(
        TARGET onnxruntime_customopregistration_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_SRC}
        $<TARGET_FILE_DIR:onnxruntime_customopregistration_test>/testdata)
    endif()

    if (UNIX AND (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV))
        # The test_main.cc includes NvInfer.h where it has many deprecated declarations
        # simply ignore them for TensorRT EP build
        set_property(TARGET onnxruntime_customopregistration_test APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    endif()

  endif()
endif()

# Build custom op library that returns an error OrtStatus when the exported RegisterCustomOps function is called.
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
  onnxruntime_add_shared_library_module(custom_op_invalid_library
                                        ${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.h
                                        ${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.cc)
  target_include_directories(custom_op_invalid_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.def")
  endif()

  set_property(TARGET custom_op_invalid_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG})
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))

  file(GLOB_RECURSE custom_op_get_const_input_test_library_src
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op.h"
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op.cc"
  )

  onnxruntime_add_shared_library_module(custom_op_get_const_input_test_library ${custom_op_get_const_input_test_library_src})

  onnxruntime_add_include_to_target(custom_op_get_const_input_test_library onnxruntime_common GTest::gtest GTest::gmock)
  target_include_directories(custom_op_get_const_input_test_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session
                                                                            ${REPO_ROOT}/include/onnxruntime/core/common)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.def")
  endif()

  set_property(TARGET custom_op_get_const_input_test_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG})
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))

  file(GLOB_RECURSE custom_op_local_function_test_library_src
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.h"
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/dummy_gemm.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/dummy_gemm.h"
  )

  onnxruntime_add_shared_library_module(custom_op_local_function ${custom_op_local_function_test_library_src})

  onnxruntime_add_include_to_target(custom_op_local_function onnxruntime_common GTest::gtest GTest::gmock)
  target_include_directories(custom_op_local_function PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session
                                                                            ${REPO_ROOT}/include/onnxruntime/core/common)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.def")
  endif()

  set_property(TARGET custom_op_local_function APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG})
endif()

# Build library that can be used with RegisterExecutionProviderLibrary and automatic EP selection
# We need a shared lib build to use that as a dependency for the test library
# Currently we only have device discovery on Windows so no point building the test app on other platforms.
if (WIN32 AND onnxruntime_BUILD_SHARED_LIB AND
    NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND
    NOT onnxruntime_MINIMAL_BUILD)
  file(GLOB onnxruntime_autoep_test_library_src "${TEST_SRC_DIR}/autoep/library/*.h"
                                                "${TEST_SRC_DIR}/autoep/library/*.cc")
  onnxruntime_add_shared_library_module(example_plugin_ep ${onnxruntime_autoep_test_library_src})
  target_include_directories(example_plugin_ep PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(example_plugin_ep PRIVATE onnxruntime)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/autoep/library/example_plugin_ep_library.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/autoep/library/example_plugin_ep_library.def")
  endif()

  set_property(TARGET example_plugin_ep APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG})

  # test library
  file(GLOB onnxruntime_autoep_test_SRC "${ONNXRUNTIME_AUTOEP_TEST_SRC_DIR}/*.h"
                                        "${ONNXRUNTIME_AUTOEP_TEST_SRC_DIR}/*.cc")

  set(onnxruntime_autoep_test_LIBS onnxruntime_mocked_allocator ${ONNXRUNTIME_TEST_LIBS} onnxruntime_test_utils
                                   onnx_proto onnx ${onnxruntime_EXTERNAL_LIBRARIES})

  if (onnxruntime_USE_TENSORRT)
    list(APPEND onnxruntime_autoep_test_LIBS ${TENSORRT_LIBRARY_INFER})
  endif()

  if (onnxruntime_USE_CUDA)
    list(APPEND onnxruntime_autoep_test_LIBS CUDA::cudart)
  endif()

  if (onnxruntime_USE_DML)
    list(APPEND onnxruntime_autoep_test_LIBS d3d12.lib)
  endif()

  if (CPUINFO_SUPPORTED)
    list(APPEND onnxruntime_autoep_test_LIBS cpuinfo)
  endif()

  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND onnxruntime_autoep_test_LIBS onnxruntime_graph onnxruntime_session onnxruntime_providers
                onnxruntime_optimizer onnxruntime_mlas onnxruntime_framework onnxruntime_util onnxruntime_flatbuffers
                iconv re2 onnx)
  endif()

  AddTest(DYN
          TARGET onnxruntime_autoep_test
          SOURCES ${onnxruntime_autoep_test_SRC} ${onnxruntime_unittest_main_src}
          LIBS ${onnxruntime_autoep_test_LIBS}
          DEPENDS ${all_dependencies} example_plugin_ep
  )
endif()

if (onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  set (onnxruntime_logging_apis_test_SRC
       ${ONNXRUNTIME_LOGGING_APIS_TEST_SRC_DIR}/test_logging_apis.cc)

  set(onnxruntime_logging_apis_test_LIBS onnxruntime_common onnxruntime_test_utils)
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND onnxruntime_logging_apis_test_LIBS onnxruntime_session onnxruntime_util onnxruntime_lora onnxruntime_framework onnxruntime_common onnxruntime_graph  onnxruntime_providers onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 ${PROTOBUF_LIB} onnx onnx_proto)
     endif()

  if(NOT WIN32)
    list(APPEND onnxruntime_logging_apis_test_LIBS  ${CMAKE_DL_LIBS})
  endif()

  AddTest(DYN
          TARGET onnxruntime_logging_apis_test
          SOURCES ${onnxruntime_logging_apis_test_SRC}
          LIBS ${onnxruntime_logging_apis_test_LIBS}
          DEPENDS ${all_dependencies}
  )
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_USE_OPENVINO AND (NOT onnxruntime_MINIMAL_BUILD OR
                                                                        onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
  onnxruntime_add_shared_library_module(custom_op_openvino_wrapper_library
                                        ${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.cc
                                        ${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/openvino_wrapper.cc)
  target_include_directories(custom_op_openvino_wrapper_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(custom_op_openvino_wrapper_library PRIVATE openvino::runtime)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.def")
  endif()

  set_property(TARGET custom_op_openvino_wrapper_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG})
endif()

# limit to only test on windows first, due to a runtime path issue on linux
if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD
                                  AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS|visionOS|tvOS"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  file(GLOB_RECURSE test_execution_provider_srcs
    "${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/*.h"
    "${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  onnxruntime_add_shared_library_module(test_execution_provider ${test_execution_provider_srcs})
  add_dependencies(test_execution_provider onnxruntime_providers_shared onnx ${ABSEIL_LIBS})
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    target_link_options(test_execution_provider PRIVATE -Wl,-brtl -lonnxruntime_providers_shared)
    target_link_libraries(test_execution_provider PRIVATE ${ABSEIL_LIBS} Boost::mp11)
  else()
    target_link_libraries(test_execution_provider PRIVATE onnxruntime_providers_shared ${ABSEIL_LIBS} Boost::mp11)
  endif()
  target_include_directories(test_execution_provider PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(test_execution_provider PRIVATE $<TARGET_PROPERTY:onnxruntime_common,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(test_execution_provider PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${ORTTRAINING_ROOT})
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    target_link_libraries(test_execution_provider PRIVATE Python::Python)
  endif()
  if(APPLE)
    set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/exported_symbols.lst")
  elseif(UNIX)
    if (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/version_script.lds -Xlinker --gc-sections -Xlinker -rpath=\\$ORIGIN")
     endif()
  elseif(WIN32)
    set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/symbols.def")
  else()
    message(FATAL_ERROR "test_execution_provider unknown platform, need to specify shared library exports for it")
  endif()
endif()

if (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EXTERNAL_DAWN)
  AddTest(TARGET onnxruntime_webgpu_external_dawn_test
          SOURCES ${onnxruntime_webgpu_external_dawn_test_SRC}
          LIBS dawn::dawn_native ${onnxruntime_test_providers_libs}
          DEPENDS ${all_dependencies}
  )
  onnxruntime_add_include_to_target(onnxruntime_webgpu_external_dawn_test dawn::dawncpp_headers dawn::dawn_headers)
endif()

if (onnxruntime_USE_WEBGPU AND WIN32 AND onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  AddTest(DYN
          TARGET onnxruntime_webgpu_delay_load_test
          SOURCES ${onnxruntime_webgpu_delay_load_test_SRC}
          LIBS ${SYS_PATH_LIB}
          DEPENDS ${all_dependencies}
  )
endif()

# onnxruntime_ep_graph_test tests the implementation of the public OrtGraph APIs for use in plugin EPs (OrtEp).
if (onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_ep_graph_test_SRC "${ONNXRUNTIME_EP_GRAPH_TEST_SRC_DIR}/*.h"
                                                  "${ONNXRUNTIME_EP_GRAPH_TEST_SRC_DIR}/*.cc")

  set(onnxruntime_ep_graph_test_LIBS ${ONNXRUNTIME_TEST_LIBS} onnxruntime_test_utils ${onnxruntime_EXTERNAL_LIBRARIES})
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND onnxruntime_ep_graph_test_LIBS onnxruntime_session onnxruntime_util onnxruntime_lora onnxruntime_framework
                                               onnxruntime_common onnxruntime_graph onnxruntime_providers onnxruntime_mlas
                                               onnxruntime_optimizer onnxruntime_flatbuffers iconv re2
                                               ${PROTOBUF_LIB} onnx onnx_proto)
  endif()

  if(NOT WIN32)
    list(APPEND onnxruntime_ep_graph_test_LIBS  ${CMAKE_DL_LIBS})
  endif()

  if (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV)
    # Need this because unittest_main_src defines a global nvinfer1::IBuilder variable.
    list(APPEND onnxruntime_ep_graph_test_LIBS ${TENSORRT_LIBRARY_INFER})
  endif()

  AddTest(DYN
          TARGET onnxruntime_ep_graph_test
          SOURCES ${onnxruntime_ep_graph_test_SRC} ${onnxruntime_unittest_main_src}
          LIBS ${onnxruntime_ep_graph_test_LIBS}
          DEPENDS ${all_dependencies}
  )
  if (UNIX AND (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV))
    # The test_main.cc includes NvInfer.h where it has many deprecated declarations
    # simply ignore them for TensorRT EP build
    set_property(TARGET onnxruntime_ep_graph_test APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
  endif()
endif()

include(onnxruntime_fuzz_test.cmake)
