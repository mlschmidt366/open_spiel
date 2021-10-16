# To enable C++ Torch AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH.
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  if(NOT OPEN_SPIEL_BUILD_WITH_LIBNOP)
    message(FATAL_ERROR
      "stochastic_alpha_zero_torch requires libnop (OPEN_SPIEL_BUILD_WITH_LIBNOP)")
  endif()

  add_library (stochastic_alpha_zero_torch OBJECT
    alpha_zero.h
    alpha_zero.cc
    device_manager.h
    model.h
    model.cc
    vpevaluator.h
    vpevaluator.cc
    vpnet.h
    vpnet.cc
  )
  target_include_directories (stochastic_alpha_zero_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(stochastic_torch_model_test stochastic_model_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:stochastic_alpha_zero_torch> $<TARGET_OBJECTS:tests>)
  add_test(stochastic_torch_model_test stochastic_torch_model_test)

  add_executable(stochastic_torch_vpnet_test stochastic_vpnet_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:stochastic_alpha_zero_torch> $<TARGET_OBJECTS:tests>)
  add_test(stochastic_torch_vpnet_test stochastic_torch_vpnet_test)

  target_link_libraries (stochastic_alpha_zero_torch ${TORCH_LIBRARIES})
  target_link_libraries (stochastic_torch_model_test ${TORCH_LIBRARIES})
  target_link_libraries (stochastic_torch_vpnet_test ${TORCH_LIBRARIES})
endif ()
