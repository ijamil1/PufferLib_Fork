/* SharksAndMinnows: a sample multiagent env about the sharks and minnows game. */

 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include "raylib.h"
 
 // Required struct. Only use floats!
 typedef struct {
     float perf; // Recommended 0-1 normalized single real number perf metric
     float score; // Recommended unnormalized single real number perf metric
     float episode_return; // Recommended metric: sum of agent rewards over episode
     float episode_length; // Recommended metric: number of steps of agent episode
     float n; // Required as the last field 
 } Log;
 
 typedef struct {
     Texture2D shark;
     Texture2D minnow;
 } Client;
 
 typedef struct {
     float x;
     float y;
     int ticks_since_reward;
 } Agent;

 typedef struct {
     float x;
     float y;
 } Shark;
 
 typedef struct {
     float x;
     float y;
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
 }
 
 void reset_sharks(SharksAndMinnows* env) {
     // reset location of all sharks
    for (int s = 0; s < env->num_sharks; s++) {
        int unique = 0;
        while (!unique) {
            int x = rand() % env->width;
            int y = rand() % (env->height - 1); // y != env->height - 1 because minnows start there

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
         for (int a=0; a<env->num_minnows; a++) {
             Agent* other = &env->minnows[a];
             env->observations[obs_idx++] = (other->x - minnow->x)/env->width;
             env->observations[obs_idx++] = (other->y - minnow->y)/env->height;
         }
         env->observations[obs_idx++] = env->rewards[m];
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
            // Check for uniqueness with previously placed sharks
            for (int t = 0; t < m; t++) {
                if (env->minnows[t].x == x && env->minnows[t].y == y) {
                    unique = 0;
                    break;
                }
            }
            if (unique) {
                env->minnows[m].x = x;
                env->minnows[m].y = y;
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
        }
    }
 }

 void update_rewards(SharksAndMinnows* env) {
     //depending on the location of the minnow and the shark, update the reward
     for (int m=0; m<env->num_minnows; m++) {
        Agent* minnow = &env->minnows[m];
        int collision = 0;
        for (int s=0; s<env->num_sharks; s++) {
            Shark* shark = &env->sharks[s];
            if (minnow->x == shark->x && minnow->y == shark->y) {
                env->rewards[m] = -1.0f;
                env->log.perf -= 1.0f;
                env->log.score -= 1.0f;
                env->log.episode_length += minnow->ticks_since_reward;
                minnow->ticks_since_reward = 0;
                env->log.episode_return -= 1.0f;
                env->log.n++;
                collision = 1;
                break
            }
        }
        if (collision) {
            reset_single_minnow(env, m);
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
            reset_single_minnow(env, m);
            reset_sharks(env);
        }
     }
 }



 // Required function
 void c_reset(SharksAndMinnows* env) {
     reset_minnows(env);
     reset_sharks(env);
     compute_observations(env);
 }

 void clip(float val, float min, float max) {
     if (val < min) {
         return min
     } else if (val > max) {
         return max
     }
     return val; 
 }
 
 
 // Required function
 void c_step(SharksAndMinnows* env) {
     for (int m=0; m<env->num_minnows; m++) {
         env->rewards[m] = 0;
         Agent* minnow = &env->minnows[m];
         minnow->ticks_since_reward += 1;
 
        if (env->actions[m] == 0) {
            //do nothing
            continue
        } else if (env->actions[m] == 1) {
            //move up towards the end of the grid (ie: y decreases)
            minnow->y -= 1;
        } else if (env->actions[m] == 2) {
            //move right (ie: x increases)
            minnow->x += 1;
        } else if (env->actions[m] == 3) {
            // move down (ie: y increases)
            minnow->y += 1;
        } else {
            // move left (ie: x decreases)
            minnow->x -= 1;
        }
                 
        minnow->x = clip(minnow->x, 0, env->width);
        minnow->y = clip(minnow->y, 0, env->height);
    
     update_rewards(env);
     compute_observations(env);
    }
}
 
 // Required function. Should handle creating the client on first call
 void c_render(SharksAndMinnows* env) {
     if (env->client == NULL) {
         InitWindow(env->width, env->height, "PufferLib Sharks and Minnows");
         SetTargetFPS(60);
         env->client = (Client*)calloc(1, sizeof(Client));
 
         // Don't do this before calling InitWindow
         env->client->minnow = LoadTexture("resources/shared/puffers_128.png");
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
             shark->x - 32,
             shark->y - 32,
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
     free(env->minnow);
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
 