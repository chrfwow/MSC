is_set = False


def init_clang():
    global is_set
    if is_set:
        return
    is_set = True
    import clang.cindex
    clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-14.so.1')
