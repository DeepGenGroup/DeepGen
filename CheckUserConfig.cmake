# CheckProgram.cmake

function(check_user_program VAR_NAME USER_PATH PROGRAM_NAME)
    if(EXISTS "${USER_PATH}" AND NOT IS_DIRECTORY "${USER_PATH}")
        get_filename_component(SEARCH_HINT "${USER_PATH}" DIRECTORY)
    else()
        set(SEARCH_HINT "${USER_PATH}")
    endif()
    # 1. 尝试寻找程序
    # HINTS 优先查找用户给出的路径
    # DOC 则是该变量在 CMakeCache.txt 中的描述
    find_program(${VAR_NAME}
        NAMES ${PROGRAM_NAME}
        HINTS ${SEARCH_HINT}
        DOC "Path to the ${PROGRAM_NAME} executable"
        NO_DEFAULT_PATH # 强制只在用户路径下找，如果想兼容系统路径可删掉此行
    )

    # 2. 验证结果
    if(${VAR_NAME})
        message(STATUS "[SUCCESS] Found ${PROGRAM_NAME} at: ${${VAR_NAME}}")
    else()
        message(FATAL_ERROR 
            "[ERROR] Could not find '${PROGRAM_NAME}' in path: ${USER_PATH}\n"
            "Please check your input variable or environment settings."
        )
    endif()
    
    # 3. 将结果同步到父级作用域，确保函数外可用
    set(${VAR_NAME} "${${VAR_NAME}}" PARENT_SCOPE)
endfunction()