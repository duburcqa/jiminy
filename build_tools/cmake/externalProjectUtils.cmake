# Import external project utilities
include(ExternalProject)

# SUBDIRSLIST
# -----------
#
# GET LIST OF SUBDIRECTORIES.
#
# dir INPUT DIRECTORY.
# surdir_list LIST OF SUBDIRECTORIES.
#
MACRO(SUBDIRLIST dir surdir_list)
  file(GLOB child_dir_list RELATIVE ${dir} ${dir}/*)
  set(dir_list "")
  foreach(child_dir ${child_dir_list})
    if(IS_DIRECTORY ${dir}/${child_dir})
      list(APPEND dir_list ${child_dir})
    endif()
	endforeach()
  set(${surdir_list} ${dir_list})
ENDMACRO()

# EXTERNALPROJECT_APPLY_PATCH
# ---------------------------------
#
# Build ExternalProject_Add patch arguments.
#
# patch_dir Path to directory containing patch files.
# OUTPUT_DOWNLOAD_ARGS Output download arguments.
#

FUNCTION(EXTERNALPROJECT_APPLY_PATCH patch_dir output_patch_args)
    set(patch_args)

	# Start patch command.
	list(APPEND patch_args PATCH_COMMAND)
	# Get all patch files in patch directory.
	file(GLOB PATCH_FILES ${patch_dir}/patch-*)
	list(SORT PATCH_FILES)
	# For each path file, append a patch instruction.
	foreach(PATCH_FILE ${PATCH_FILES})
		list(APPEND patch_args
			patch -p0 < ${PATCH_FILE} &&
		)
	endforeach()
	list(REMOVE_AT patch_args -1)

	set(${output_patch_args} "${patch_args}" PARENT_SCOPE)
ENDFUNCTION ()