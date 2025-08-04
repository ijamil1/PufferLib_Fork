#include "sharks_and_minnows.h"

#define Env SharksAndMinnows
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_minnows = unpack(kwargs, "num_minnows");
    env->num_sharks = unpack(kwargs, "num_sharks");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "shark_collisions", log->shark_collisions);
    assign_to_dict(dict, "minnow_goal_reaches", log->minnow_goal_reaches);
    assign_to_dict(dict, "left_moves", log->left_moves);
    assign_to_dict(dict, "right_moves", log->right_moves);
    assign_to_dict(dict, "up_moves", log->up_moves);
    assign_to_dict(dict, "down_moves", log->down_moves);
    assign_to_dict(dict, "stay_moves", log->stay_moves);
    assign_to_dict(dict, "left_upwards_diagonal_moves", log->left_upwards_diagonal_moves);
    assign_to_dict(dict, "right_upwards_diagonal_moves", log->right_upwards_diagonal_moves);
    assign_to_dict(dict, "left_downwards_diagonal_moves", log->left_downwards_diagonal_moves);
    assign_to_dict(dict, "right_downwards_diagonal_moves", log->right_downwards_diagonal_moves);
    return 0;
}
