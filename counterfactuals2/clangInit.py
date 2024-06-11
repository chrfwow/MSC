is_set = False


def init_clang():
    global is_set
    if is_set:
        return
    is_set = True
    import clang.cindex
    clang.cindex.Config.set_library_file('D:/Programme/LLVM/bin/libclang.dll')
