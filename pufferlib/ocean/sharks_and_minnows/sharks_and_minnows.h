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
     float left_upwards_diagonal_moves;
     float right_upwards_diagonal_moves;
     float left_downwards_diagonal_moves;
     float right_downwards_diagonal_moves;
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
     int direction;
     int ticks_since_reward;
 } Agent;

 typedef struct {
     int x;
     int y;
     int direction;
     int paused;
     int ticks_since_pause;
     int minnow_target;
 } Shark;
 
 typedef struct {
     int x;
     int y;
 } Goal;

typedef enum { STAY=0, UP=1, RIGHT=2, DOWN=3, LEFT=4, UP_LEFT=5, UP_RIGHT=6, DOWN_LEFT=7, DOWN_RIGHT=8 } Direction;

 
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
    env->log.left_upwards_diagonal_moves = 0.0f;
    env->log.right_upwards_diagonal_moves = 0.0f;
    env->log.left_downwards_diagonal_moves = 0.0f;
    env->log.right_downwards_diagonal_moves = 0.0f;
    env->log.shark_collisions = 0.0f;
    env->log.minnow_goal_reaches = 0.0f;
    env->log.episode_return = 0.0f;
    env->log.episode_length = 0.0f;
    env->log.n = 0.0f;
 }
 
 void reset_sharks(SharksAndMinnows* env) {
     // reset location of all sharks
    int num_sharks = env->num_sharks;
    int width = env->width;
    for (int s = 0; s < env->num_sharks; s++) {
        int unique = 0;
        while (!unique) {
            int x = rand() % (env->width/num_sharks) + (s * (env->width/num_sharks));
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
                env->sharks[s].direction = rand() % 4;
                env->sharks[s].paused = 0;
                env->sharks[s].ticks_since_pause = 0;
                env->sharks[s].minnow_target = -1;
            }
        }
    }
 }

 void pause_single_shark(SharksAndMinnows* env, int s) {
    env->sharks[s].paused = 1;
    env->sharks[s].minnow_target = -1;
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
             env->observations[obs_idx++] = (shark->minnow_target == m);
         }
         for (int a=0; a<env->num_minnows; a++) {
            if (a == m) {
                continue;
            }
             Agent* other = &env->minnows[a];
             env->observations[obs_idx++] = (other->x - minnow->x)/env->width;
             env->observations[obs_idx++] = (other->y - minnow->y)/env->height;
         }
         env->observations[obs_idx++] = minnow->x/env->width;
         env->observations[obs_idx++] = minnow->y/env->height;
     }
 }

 
 void reset_minnows(SharksAndMinnows* env) {
     //reset location of all minnows
    int num_minnows = env->num_minnows;
    int width = env->width;
     for (int m = 0; m < env->num_minnows; m++) {
        int unique = 0;
        while (!unique) {
            int x = rand() % (env->width/num_minnows) + (m * (env->width/num_minnows));
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
            env->minnows[m].ticks_since_reward = 0;  // Reset episode counter
        }
    
    }
 }

 void update_rewards(SharksAndMinnows* env) {
    const float GOAL_REWARD = 1.0f;
    const float UPWARD_REWARD = 0.005f;
    const float DANGER_RADIUS = 100.0f;
    const float CAPTURE_RADIUS = 25.0f;
    const float OPTIMAL_RESPONSE_REWARD = 0.1f;
    const float OPTIMAL_RESPONSE_PENALTY = 0.1f;
    const float SURVIVAL_REWARD = 0.01f;
    const float CAPTURE_PENALTY = 0.40f;
    char shark_capture_ind[env->num_sharks];
    
    for (int i = 0; i < env->num_sharks; i++) {
        shark_capture_ind[i] = 0;
    }
   
    for (int m = 0; m < env->num_minnows; m++) {
        Agent* minnow = &env->minnows[m];
        float reward = 0.0f;
        float min_chaser_dist = 1e9;
        int captured = 0;
        int shark_direction = STAY;
        int shark_x = 0;
        int shark_y = 0;
        int minnow_direction = minnow->direction;

        for (int s = 0; s < env->num_sharks; s++) {
            Shark* shark = &env->sharks[s];
            float dx = minnow->x - shark->x;
            float dy = minnow->y - shark->y;
            float dist = sqrt(dx * dx + dy * dy);

            if (shark->minnow_target == m) {
                shark_direction = shark->direction;
                shark_x = shark->x;
                shark_y = shark->y;
                min_chaser_dist = dist;
            }
            if (dist <= CAPTURE_RADIUS) {
                // captured
                if (captured == 0) {
                    reward -= CAPTURE_PENALTY; 
                    env->log.perf -= CAPTURE_PENALTY;
                    env->log.shark_collisions += 1.0f;
                }
                captured = 1;
                shark_capture_ind[s] = 1;
            }
        }

        if (captured == 0 && minnow->y <= 0.0f) {
            // Terminal event: reached goal
            reward = GOAL_REWARD;
            env->terminals[m] = 1;
            env->log.perf += GOAL_REWARD;
            env->log.score += GOAL_REWARD;
            env->log.episode_length += minnow->ticks_since_reward;
            env->log.episode_return += GOAL_REWARD;
            env->log.minnow_goal_reaches += 1.0f;
            env->log.n++;
            minnow->ticks_since_reward = 0;
            env->rewards[m] = reward;
            continue;
        }

        // --- Shaping rewards ---

        if (captured == 0) {
            //if not captured, give survival reward
            reward += SURVIVAL_REWARD;

            // if not captured, give upward reward if minnow is moving upward
            if ((minnow_direction == UP || minnow_direction == UP_LEFT || minnow_direction == UP_RIGHT) && min_chaser_dist > DANGER_RADIUS && minnow->y % 5 == 0) {
                reward += UPWARD_REWARD;
            }
        
            bool is_optimal = false;

            switch (shark_direction) {
                case LEFT:
                    is_optimal = (minnow_direction == UP_LEFT || minnow_direction == LEFT || minnow_direction == DOWN_LEFT);  // moving left
                    break;
                case RIGHT:
                    is_optimal = (minnow_direction == UP_RIGHT || minnow_direction == RIGHT || minnow_direction == DOWN_RIGHT);   // moving right
                    break;
                case UP:
                    is_optimal = (minnow_direction != DOWN && minnow_direction != DOWN_LEFT && minnow_direction != DOWN_RIGHT); // any upward or lateral motion
                    break;
                case DOWN:
                    if (minnow->x < shark_x) {
                        // if minnow is to the left of shark
                        is_optimal = (minnow_direction == UP_LEFT || minnow_direction == UP || minnow_direction == LEFT || minnow_direction == DOWN_LEFT || minnow_direction == DOWN);
                    }
                    else if (minnow->x > shark_x) {
                        // if minnow is to the right of shark
                        is_optimal = (minnow_direction == UP_RIGHT || minnow_direction == UP || minnow_direction == RIGHT || minnow_direction == DOWN_RIGHT || minnow_direction == DOWN);
                    }
                    else {
                        // if minnow is directly below shark, it can move up or down
                        is_optimal = (minnow_direction != UP); 
                    }
                    break;
                case STAY:
                    if (minnow->x < shark_x) {
                        // if minnow is to the left of shark
                        is_optimal = (minnow_direction == UP_LEFT || minnow_direction == UP || minnow_direction == LEFT || minnow_direction == DOWN_LEFT || minnow_direction == DOWN);
                    }
                    else if (minnow->x > shark_x) {
                        // if minnow is to the right of shark
                        is_optimal = (minnow_direction == UP_RIGHT || minnow_direction == UP || minnow_direction == RIGHT || minnow_direction == DOWN_RIGHT || minnow_direction == DOWN);
                    }
                    else {
                        // if minnow is directly above/below shark, it can move up or down depnding on the pos of the shark
                        if (shark_y < minnow->y) {
                            // if shark is above minnow
                            is_optimal = (minnow_direction != UP);
                        }
                        else {
                            // if shark is below minnow
                            is_optimal = (minnow_direction != DOWN);
                        }
                    }
            }

            if (is_optimal && min_chaser_dist <= DANGER_RADIUS) {
                reward += OPTIMAL_RESPONSE_REWARD;
            }
            else if (min_chaser_dist <= DANGER_RADIUS) {
                reward -= OPTIMAL_RESPONSE_PENALTY;
            }
        }

        // --- Bookkeeping ---
        env->rewards[m] = reward;
        env->log.score += reward;
        env->log.episode_return += reward;
    }

    // need to reset sharks that captured
    for (int s = 0; s < env->num_sharks; s++) {
        if (shark_capture_ind[s]) {
            pause_single_shark(env, s);
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

int wrap(int val, int min, int max) {
    if (val < min) {
        return max;
    } else if (val > max) {
        return min;
    }
    return val;
}


 
 // Add this helper function at an appropriate place (e.g., after clip)
float distance(int x1, int y1, int x2, int y2) {
    return sqrtf(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}


// Add this function to move sharks toward closest minnow
void move_sharks_toward_minnows(SharksAndMinnows* env) {
    int shark_direction = STAY;
    int sharks_set = 0;

    char minnows_chased[env->num_minnows];
    for (int i = 0; i < env->num_minnows; i++) {
        minnows_chased[i] = 0;
    }

    float dirs[5][2] = {
        {0, 0},    // stay
        {0, -1},   // up
        {1, 0},    // right
        {0, 1},    // down
        {-1, 0}    // left
    };

  
    for (int s = 0; s < env->num_sharks; s++) {
        if (env->sharks[s].paused && env->sharks[s].ticks_since_pause < 5) {
            env->sharks[s].ticks_since_pause += 1;
            env->sharks[s].direction = STAY;
            continue;
        }
        else if (env->sharks[s].paused && env->sharks[s].ticks_since_pause >= 5) {
            env->sharks[s].paused = 0;
            env->sharks[s].ticks_since_pause = 0;
        }
        
        Shark* shark = &env->sharks[s];
        shark_direction = STAY;
        // Find closest minnow
        int minnow_target = -1;
        float min_dist = 1e9;
        float minnow_x = 0, minnow_y = 0;
        for (int m = 0; m < env->num_minnows; m++) {
            if (minnows_chased[m]) {
                continue;
            }
            Agent* minnow = &env->minnows[m];
            float d = distance(shark->x, shark->y, minnow->x, minnow->y);
            if (d < min_dist) {
                min_dist = d;
                minnow_x = minnow->x;
                minnow_y = minnow->y;
                minnow_target = m;
            }
        }
        minnows_chased[minnow_target] = 1;
        // Try all possible directions: 0=stay, 1=up, 2=right, 3=down, 4=left
        float best_x = shark->x, best_y = shark->y;
        float best_dist = distance(shark->x, shark->y, minnow_x, minnow_y);
        
        for (int d = 1; d < 5; d++) {
            float nx = shark->x + dirs[d][0];
            float ny = shark->y + dirs[d][1];
            nx = wrap(nx, 0, env->width - 1);
            ny = clip(ny, 1, env->height - 2);
            float d_to_m = distance(nx, ny, minnow_x, minnow_y);
            if (ny == env->height - 1 || ny == 0) {
                d_to_m = 1e9;
            }
            if (d_to_m < best_dist) {
                shark_direction = d;
                best_dist = d_to_m;
                best_x = nx;
                best_y = ny;
            }
        }
        shark->x = best_x;
        shark->y = best_y;
        shark->direction = shark_direction;
        shark->minnow_target = minnow_target;
    }
}
 
 
 // Required function
 void c_step(SharksAndMinnows* env) {
    check_shark_positions(env);
    check_minnow_positions(env);
    move_sharks_toward_minnows(env); // sharks move first, toward distinct minnows

  
    for (int m=0; m<env->num_minnows; m++) {
         env->rewards[m] = 0;
         if (env->terminals[m]) {
            //if minnow is in a terminal state, reset the state and reset the terminal flag and move on to the next minnow
            reset_single_minnow(env, m);
            env->terminals[m] = 0;
            continue;
         }
         env->terminals[m] = 0;
         Agent* minnow = &env->minnows[m];
         minnow->ticks_since_reward += 1;
         
 
        if (env->actions[m] == 0) {
            //do nothing
            env->log.stay_moves += 1.0f;
            minnow->direction = STAY;
            continue;
        } else if (env->actions[m] == 1) {
            //move up towards the end of the grid (ie: y decreases)
            env->log.up_moves += 1.0f;
            minnow->y -= 1;
            minnow->direction = UP;
        } else if (env->actions[m] == 2) {
            //move right (ie: x increases)
            env->log.right_moves += 1.0f;
            minnow->x += 1;
            minnow->direction = RIGHT;
        } else if (env->actions[m] == 3) {
            // move down (ie: y increases)
            env->log.down_moves += 1.0f;
            minnow->y += 1;
            minnow->direction = DOWN;
        } else if (env->actions[m] == 4) {
            // move left (ie: x decreases)
            env->log.left_moves += 1.0f;
            minnow->x -= 1;
            minnow->direction = LEFT;
        } else if (env->actions[m] == 5) {
            // move left upwards diagonally (ie: x decreases, y decreases)
            env->log.left_upwards_diagonal_moves += 1.0f;
            minnow->x -= 1;
            minnow->y -= 1;
            minnow->direction = UP_LEFT;
        } else if (env->actions[m] == 6) {
            // move right upwards diagonally (ie: x increases, y decreases)
            env->log.right_upwards_diagonal_moves += 1.0f;
            minnow->x += 1;
            minnow->y -= 1;
            minnow->direction = UP_RIGHT;
        } else if (env->actions[m] == 7) {
            // move left downwards diagonally (ie: x decreases, y increases)
            env->log.left_downwards_diagonal_moves += 1.0f;
            minnow->x -= 1;
            minnow->y += 1;
            minnow->direction = DOWN_LEFT;
        } else if (env->actions[m] == 8) {
            // move right downwards diagonally (ie: x increases, y increases)
            env->log.right_downwards_diagonal_moves += 1.0f;
            minnow->x += 1;
            minnow->y += 1;
            minnow->direction = DOWN_RIGHT;
        }
        
                 
        minnow->x = wrap(minnow->x, 0, env->width - 1);
        minnow->y = clip(minnow->y, 0, env->height - 1); 

        if (minnow->ticks_since_reward % (env->height * 4) == 0) {
            minnow->x = rand() % env->width;
            minnow->y = env->height - 1;
        }
    }

    update_rewards(env); //for the minnows that were in a terminal state and got reset, the reward will be 0
    compute_observations(env);
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
 