import os.path

is_set = False


def init_clang():
    global is_set
    if is_set:
        return

    fname = 'D:/Programme/LLVM/bin/libclang.dll'
    if not os.path.isfile(fname):
        print(fname, "is not a file")
        return

    print("setting clang lib file")
    import clang.cindex
    clang.cindex.Config.set_library_file(fname)
    is_set = True

    print(fname, "set as clang lib file")
