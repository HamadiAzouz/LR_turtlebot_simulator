# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_lr_ppo_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED lr_ppo_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(lr_ppo_FOUND FALSE)
  elseif(NOT lr_ppo_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(lr_ppo_FOUND FALSE)
  endif()
  return()
endif()
set(_lr_ppo_CONFIG_INCLUDED TRUE)

# output package information
if(NOT lr_ppo_FIND_QUIETLY)
  message(STATUS "Found lr_ppo: 1.0.0 (${lr_ppo_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'lr_ppo' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT lr_ppo_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(lr_ppo_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${lr_ppo_DIR}/${_extra}")
endforeach()
