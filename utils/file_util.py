import os
import shutil


def backup_code(dir_dst_root):
    dir_sec = os.getcwd()
    dir_dst = os.path.join(dir_dst_root, 'src')
    copy_floder_dfs(dir_sec, dir_dst)


def copy_floder_dfs(dir_src, dir_dst, depth=0, filter=['.py']):
    # If the dir_dst is in dir_src, the max_depth is very important, if it set wrong will fall into an infinite loop.
    if depth >= 3:
        return
    if not os.path.exists(dir_dst):
        os.makedirs(dir_dst)

    names = os.listdir(dir_src)
    for name in names:
        path_src = os.path.join(dir_src, name)
        path_dst = os.path.join(dir_dst, name)
        if os.path.isfile(path_src):
            if filter.count(os.path.splitext(name)[-1]):
                shutil.copy(path_src, path_dst)
        else:
            copy_floder_dfs(path_src, path_dst, depth+1)