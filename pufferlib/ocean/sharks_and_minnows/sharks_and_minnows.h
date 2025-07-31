/* SharksAndMinnows: a sample multiagent env about the sharks and minnows game. */

 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include "raylib.h"
 #include <stdio.h>
 #include <assert.h>
 
 // Required struct. Only use floats!
 typedef struct {
     float perf; // Recommended 0-1 normalized single real number perf metric
     float score; // Recommended unnormalized single real number perf metric
     float shark_collisions;
     float minnow_goal_reaches;
     float left_moves;
     float right_moves;
     float up_moves;
     float down_moves;
     float stay_moves;
     float episode_return; // Recommended metric: sum of agent rewards over episode
     float episode_length; // Recommended metric: number of steps of agent episode
     float n; // Required as the last field 
 } Log;
 
 typedef struct {
     Texture2D shark;
     Texture2D minnow;
 } Client;
 
 typedef struct {
     int x;
     int y;
     int prev_y;
     int ticks_since_reward;
 } Agent;

 typedef struct {
     int x;
     int y;
 } Shark;
 
 typedef struct {
     int x;
     int y;
 } Goal;
 
 // Required that you have some struct for your env
 // Recommended that you name it the same as the env file
 typedef struct {
     Log log; // Required field. Env binding code uses this to aggregate logs
     Client* client;
     Agent* minnows;
     Shark* sharks;
     Goal* goal;
     float* observations; // Required. You can use any obs type, but make sure it matches in Python!
     int* actions; // Required. int* for discrete/multidiscrete, float* for box
     float* rewards; // Required
     unsigned char* terminals; // Required. We don't yet have truncations as standard yet
     int width;
     int height;
     int num_minnows;
     int num_sharks;
 } SharksAndMinnows;
 
 /* Recommended to have an init function of some kind if you allocate 
  * extra memory. This should be freed by c_close. Don't forget to call
  * this in binding.c!
  */
 void init(SharksAndMinnows* env) {

    if (env->num_minnows > env->width/2) {
        fprintf(stderr, "Error: num_minnows (%d) must be <= width/2 (%d)\n", env->num_minnows, env->width/2);
        exit(1);
    }

    if (env->num_sharks > env->height * env->width/2) {
        fprintf(stderr, "Error: num_sharks (%d) must be <= height * width/2 (%d)\n", env->num_sharks, env->height * env->width/2);
        exit(1);
    }
    
    env->minnows = calloc(env->num_minnows, sizeof(Agent));
    env->sharks = calloc(env->num_sharks, sizeof(Shark));
    env->goal = calloc(1, sizeof(Goal));
    
    // Explicitly initialize ticks_since_reward to 0 for all minnows
    for (int m = 0; m < env->num_minnows; m++) {
        env->minnows[m].ticks_since_reward = 0;
    }
    env->log.left_moves = 0.0f;
    env->log.right_moves = 0.0f;
    env->log.up_moves = 0.0f;
    env->log.down_moves = 0.0f;
    env->log.stay_moves = 0.0f;
    env->log.shark_collisions = 0.0f;
    env->log.minnow_goal_reaches = 0.0f;
    env->log.episode_return = 0.0f;
    env->log.episode_length = 0.0f;
    env->log.n = 0.0f;
 }
 
 void reset_sharks(SharksAndMinnows* env) {
     // reset location of all sharks
    for (int s = 0; s < env->num_sharks; s++) {
        int unique = 0;
        while (!unique) {
            int x = rand() % env->width;
            int y = rand() % (env->height/2); // sharks start in the top half of the grid

            unique = 1;
            // Check for uniqueness with previously placed sharks
            for (int t = 0; t < s; t++) {
                if (env->sharks[t].x == x && env->sharks[t].y == y) {
                    unique = 0;
                    break;
                }
            }
            for (int t = 0; t < env->num_minnows; t++) {
                if (env->minnows[t].x == x && env->minnows[t].y == y) {
                    unique = 0;
                    break;
                }
            }
            if (y <= 0 || y >= env->height - 1 || x < 0 || x > env->width - 1) {
                unique = 0;
            }
            if (unique) {
                env->sharks[s].x = x;
                env->sharks[s].y = y;
            }
        }
    }
 }
 
 /* Recommended to have an observation function of some kind because
  * you need to compute agent observations in both reset and in step.
  * If using float obs, try to normalize to roughly -1 to 1 by dividing
  * by an appropriate constant.
  */
 void compute_observations(SharksAndMinnows* env) {
     int obs_idx = 0;
     for (int m=0; m<env->num_minnows; m++) {
         Agent* minnow = &env->minnows[m];
         for (int s=0; s<env->num_sharks; s++) {
             Shark* shark = &env->sharks[s];
             env->observations[obs_idx++] = (shark->x - minnow->x)/env->width;
             env->observations[obs_idx++] = (shark->y - minnow->y)/env->height;
         }
         //for (int a=0; a<env->num_minnows; a++) {
         //    Agent* other = &env->minnows[a];
         //    env->observations[obs_idx++] = (other->x - minnow->x)/env->width;
         //    env->observations[obs_idx++] = (other->y - minnow->y)/env->height;
         //}
         //env->observations[obs_idx++] = env->rewards[m];
         env->observations[obs_idx++] = minnow->x/env->width;
         env->observations[obs_idx++] = minnow->y/env->height;
     }
 }
 
 void reset_minnows(SharksAndMinnows* env) {
     //reset location of all minnows
     for (int m = 0; m < env->num_minnows; m++) {
        int unique = 0;
        while (!unique) {
            int x = rand() % env->width;
            int y = env->height - 1;

            unique = 1;
            // Check for uniqueness with previously placed minnows
            for (int t = 0; t < m; t++) {
                if (env->minnows[t].x == x && env->minnows[t].y == y) {
                    unique = 0;
                    break;
                }
            }
            if (unique) {
                env->minnows[m].x = x;
                env->minnows[m].y = y;
                env->minnows[m].prev_y = y;
                env->minnows[m].ticks_since_reward = 0;  // Reset episode counter
            }
        }
    }
 }

 void reset_single_minnow(SharksAndMinnows* env, int m) {
    int unique = 0;
    while (!unique) {
        int x = rand() % env->width;
        int y = env->height - 1;

        unique = 1;
        
        for (int t = 0; t < env->num_minnows; t++) {
            if (env->minnows[t].x == x && env->minnows[t].y == y) {
                unique = 0;
                break;
            }
        }
       
        if (unique) {
            env->minnows[m].x = x;
            env->minnows[m].y = y;
            env->minnows[m].prev_y = y;
            env->minnows[m].ticks_since_reward = 0;  // Reset episode counter
        }
    
    }
 }

 void update_rewards(SharksAndMinnows* env) {
     //depending on the location of the minnow and the shark, update the reward
     for (int m=0; m<env->num_minnows; m++) {
        Agent* minnow = &env->minnows[m];
        int collision = 0;
        float min_dist = 1e9;
        for (int s=0; s<env->num_sharks; s++) {
            Shark* shark = &env->sharks[s];
            float dist = sqrt(pow(minnow->x - shark->x, 2) + pow(minnow->y - shark->y, 2));
            if (dist < min_dist) {
                min_dist = dist;
            }
            if (dist <= 48) {
                env->rewards[m] = -1.0f;
                env->log.perf -= 1.0f;
                env->log.score -= 1.0f;
                env->log.episode_length += minnow->ticks_since_reward;
                minnow->ticks_since_reward = 0;
                env->log.episode_return -= 1.0f;
                env->log.n++;
                collision = 1;
                break;
            }
        }
        if (collision) {
            env->log.shark_collisions += 1.0f;
            env->terminals[m] = 1;
            continue;
        }
        if (minnow->y == 0) {
            env->rewards[m] = 1.0f;
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += minnow->ticks_since_reward;
            minnow->ticks_since_reward = 0;
            env->log.episode_return += 1.0f;
            env->log.n++;
            env->log.minnow_goal_reaches += 1.0f;
            env->terminals[m] = 1;
        }
        else {
            if (minnow->y < minnow->prev_y) {
                env->rewards[m] = 0.005f;
                env->log.score += 0.005f;
            }
            if (min_dist <= 64 && min_dist > 48) {
                env->rewards[m] = -0.35f;
                env->log.score -= 0.35f;
            }
        }
     }
 }


 void check_shark_positions(SharksAndMinnows* env) {
    if (!env->sharks) {
        printf("Error: sharks pointer is NULL\n");
        return;
    }
    for (int s = 0; s < env->num_sharks; s++) {
        Shark* shark = &env->sharks[s];
        if (shark->x < 0 || shark->x > env->width - 1 || shark->y < 1 || shark->y > env->height - 2) {
            printf("Shark %d is out of valid range: x=%d, y=%d\n", s, shark->x, shark->y);
        }
    }
 }

 void check_minnow_positions(SharksAndMinnows* env) {
    if (!env->minnows) {
        printf("Error: minnows pointer is NULL\n");
        return;
    }
    for (int m = 0; m < env->num_minnows; m++) {
        Agent* minnow = &env->minnows[m];
        if (minnow->x < 0 || minnow->x >= env->width || minnow->y < 0 || minnow->y >= env->height) {
            printf("Minnow %d is out of valid range: x=%d, y=%d\n", m, minnow->x, minnow->y);
        }
    }
 }

 void sanity_check_starting_positions(SharksAndMinnows* env) {
    for (int m = 0; m < env->num_minnows; m++) {
        Agent* minnow = &env->minnows[m];
        assert(minnow->y == env->height - 1);
        assert(minnow->x >= 0 && minnow->x < env->width);
    }
    for (int s = 0; s < env->num_sharks; s++) {
        Shark* shark = &env->sharks[s];
        assert(shark->y >= 0 && shark->y < (env->height/2));
        assert(shark->x >= 0 && shark->x < env->width);
      
    }
 }

 // Required function
 void c_reset(SharksAndMinnows* env) {
     reset_minnows(env);
     reset_sharks(env);
     compute_observations(env);
     sanity_check_starting_positions(env);
 }


 int clip(int val, int min, int max) {
     if (val < min) {
         return min;
     } else if (val > max) {
         return max;
     }
     return val;
 }

 
 // Add this helper function at an appropriate place (e.g., after clip)
float distance(int x1, int y1, int x2, int y2) {
    return sqrtf(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// Add this function to move sharks toward closest minnow
void move_sharks_toward_minnows(SharksAndMinnows* env) {

    float dirs[5][2] = {
        {0, 0},    // stay
        {0, -1},   // up
        {1, 0},    // right
        {0, 1},    // down
        {-1, 0}    // left
    };

    for (int s = 0; s < env->num_sharks; s++) {
        float r = (float)rand() / RAND_MAX;
        Shark* shark = &env->sharks[s];
        // 60% chance to move randomly instead of chasing minnows
        if (r > 0.4) {
            int random_dir = rand() % 5; // 0-4 for stay,up,right,down,left
            shark->x = clip(shark->x + dirs[random_dir][0], 0, env->width - 1);
            shark->y = clip(shark->y + dirs[random_dir][1], 1, env->height - 2);
            continue;
        }
        // Find closest minnow
        float min_dist = 1e9;
        float minnow_x = 0, minnow_y = 0;
        for (int m = 0; m < env->num_minnows; m++) {
            Agent* minnow = &env->minnows[m];
            float d = distance(shark->x, shark->y, minnow->x, minnow->y);
            if (d < min_dist) {
                min_dist = d;
                minnow_x = minnow->x;
                minnow_y = minnow->y;
            }
        }
        // Try all possible directions: 0=stay, 1=up, 2=right, 3=down, 4=left
        float best_x = shark->x, best_y = shark->y;
        float best_dist = distance(shark->x, shark->y, minnow_x, minnow_y);
        
        for (int d = 1; d < 5; d++) {
            float nx = shark->x + dirs[d][0];
            float ny = shark->y + dirs[d][1];
            nx = clip(nx, 0, env->width - 1);
            ny = clip(ny, 1, env->height - 2);
            float d_to_m = distance(nx, ny, minnow_x, minnow_y);
            if (ny == env->height - 1 || ny == 0) {
                d_to_m = 1e9;
            }
            if (d_to_m < best_dist) {
                best_dist = d_to_m;
                best_x = nx;
                best_y = ny;
            }
        }
        shark->x = best_x;
        shark->y = best_y;
    }
}
 
 
 // Required function
 void c_step(SharksAndMinnows* env) {
    check_shark_positions(env);
    check_minnow_positions(env);
    move_sharks_toward_minnows(env); // sharks move first, toward closest minnow
    //printf("DEBUG: Sharks moved, processing minnows\n");

    for (int m=0; m<env->num_minnows; m++) {
         env->rewards[m] = 0;
         if (env->terminals[m]) {
            //if minnow is in a terminal state, reset the state and reset the terminal flag and move on to the next minnow
            //printf("DEBUG: Minnow %d is in a terminal state, resetting\n", m);
            reset_single_minnow(env, m);
            env->terminals[m] = 0;
            continue;
         }
         Agent* minnow = &env->minnows[m];
         minnow->ticks_since_reward += 1;
         minnow->prev_y = minnow->y;
 
        if (env->actions[m] == 0) {
            //do nothing
            env->log.stay_moves += 1.0f;
            continue;
        } else if (env->actions[m] == 1) {
            //move up towards the end of the grid (ie: y decreases)
            env->log.up_moves += 1.0f;
            minnow->y -= 1;
        } else if (env->actions[m] == 2) {
            //move right (ie: x increases)
            env->log.right_moves += 1.0f;
            minnow->x += 1;
        } else if (env->actions[m] == 3) {
            // move down (ie: y increases)
            env->log.down_moves += 1.0f;
            minnow->y += 1;
        } else {
            // move left (ie: x decreases)
            env->log.left_moves += 1.0f;
            minnow->x -= 1;
        }
                 
        minnow->x = clip(minnow->x, 0, env->width - 1);
        minnow->y = clip(minnow->y, 0, env->height - 1); 

        if (minnow->ticks_since_reward % (env->height * 4) == 0) {
            minnow->x = rand() % env->width;
            minnow->y = env->height - 1;
        }
    }
    //printf("DEBUG: Minnows moved\n");
    update_rewards(env); //for the minnows that were in a terminal state and got reset, the reward will be 0
    //printf("DEBUG: Rewards set\n");
    compute_observations(env);
    //printf("DEBUG: Observations computed\n");
}
 
 // Required function. Should handle creating the client on first call
 void c_render(SharksAndMinnows* env) {
     if (env->client == NULL) {
         InitWindow(env->width, env->height, "PufferLib Sharks and Minnows");
         SetTargetFPS(60);
         env->client = (Client*)calloc(1, sizeof(Client));
 
         // Don't do this before calling InitWindow
         env->client->minnow = LoadTexture("resources/sharks_and_minnows/Fish.png");
         env->client->shark = LoadTexture("resources/sharks_and_minnows/shark_icon.png");
     }
 
     // Standard across our envs so exiting is always the same
     if (IsKeyDown(KEY_ESCAPE)) {
         exit(0);
     }
 
     BeginDrawing();
     ClearBackground((Color){6, 24, 24, 255});
 
     for (int i=0; i<env->num_sharks; i++) {
         Shark* shark = &env->sharks[i];
         DrawTexture(
             env->client->shark,
             shark->x - 16,
             shark->y - 16,
             WHITE
         );
     }
 
     for (int i=0; i<env->num_minnows; i++) {
         Agent* minnow = &env->minnows[i];
         DrawTexture(
            env->client->minnow,
            minnow->x - 32,
            minnow->y - 32,
            WHITE
        );
     }
 
     EndDrawing();
 }
 
 // Required function. Should clean up anything you allocated
 // Do not free env->observations, actions, rewards, terminals
 void c_close(SharksAndMinnows* env) {
     free(env->minnows);
     free(env->sharks);
     free(env->goal);
     if (env->client != NULL) {
         Client* client = env->client;
         UnloadTexture(client->minnow);
         UnloadTexture(client->shark);
         CloseWindow();
         free(client);
     }
 }
 