#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ruckig::ruckig" for configuration "Release"
set_property(TARGET ruckig::ruckig APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ruckig::ruckig PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libruckig.so"
  IMPORTED_SONAME_RELEASE "libruckig.so"
  )

list(APPEND _cmake_import_check_targets ruckig::ruckig )
list(APPEND _cmake_import_check_files_for_ruckig::ruckig "${_IMPORT_PREFIX}/lib/libruckig.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
