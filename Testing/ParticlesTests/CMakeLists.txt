set(TEST_SRCS
  ParticlesTests.cpp
  )

add_executable(ParticlesTests
  ${TEST_SRCS}
  )

target_link_libraries(ParticlesTests
  ${ITK_LIBRARIES} ${VTK_LIBRARIES}
  Particles gtest_main)

add_test(NAME ParticlesTests COMMAND ParticlesTests)
