/* Build it with:
 * bash scripts/build_ocean.sh sharks_and_minnows local (debug)
 * bash scripts/build_ocean.sh sharks_and_minnows fast
 */

 #include "sharks_and_minnows.h"
 
 int main() {
     int num_minnows = 5;
     int num_sharks = 5;
     int num_obs = 2*(num_minnows + num_sharks) + 3;
 
 
     SharksAndMinnows env = {
         .width = 1080,
         .height = 720,
         .num_minnows = num_minnows,
         .num_sharks = num_sharks 
     };
     init(&env);
 
     // Allocate these manually since they aren't being passed from Python
     env.observations = calloc(env.num_minnows*num_obs, sizeof(float));
     env.actions = calloc(env.num_minnows, sizeof(int));
     env.rewards = calloc(env.num_minnows, sizeof(float));
     env.terminals = calloc(env.num_minnows, sizeof(unsigned char));
 
     // Always call reset and render first
     c_reset(&env);
     c_render(&env);
 
     // while(True) will break web builds
     while (!WindowShouldClose()) {
         for (int i=0; i<env.num_minnows; i++) {
             env.actions[i] = 1;
         }
 
         c_step(&env);
         c_render(&env);
     }
 
     // Try to clean up after yourself
     free(env.observations);
     free(env.actions);
     free(env.rewards);
     free(env.terminals);
     c_close(&env);
 }
 
 