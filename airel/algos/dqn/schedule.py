class LinearSchedule(object):
    def __init__(self, total_timesteps, exploration_fraction, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters:

            schedule_timesteps (int):
                Number of timesteps for which to linearly anneal initial_p
                to final_p.
            initial_p (float): Initial output value.
            final_p (float): Final output value.
        
        Example:

            ::

                >>> #TODO
        """
        
        self.schedule_timesteps = int(exploration_fraction * total_timesteps)
        self.final_p = final_p
        self.initial_p = initial_p

    def get(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)