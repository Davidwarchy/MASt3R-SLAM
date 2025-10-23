import math
import time
import numpy as np
import random

class RobotController:
    def __init__(self, robot, leftMotor, rightMotor, timestep):
        self.robot = robot
        self.leftMotor = leftMotor
        self.rightMotor = rightMotor
        self.timestep = timestep

        # Motion constants
        self.MAX_SPEED = math.pi * 2
        self.TURN_DURATION = 1.2  # seconds for a ~90° turn
        self.FORWARD_DURATION = 1.5  # seconds for a small forward step

        # Stuck detection
        self.last_check_time = time.time()
        self.last_pos = None
        self.stuck_counter = 0
        self.STUCK_LIMIT = 10
        self.MIN_MOVE_DIST = 0.05  # meters (tune as needed)
        self.CHECK_INTERVAL = 5.0  # seconds

        # Possible movements for random sequence
        self.POSSIBLE_MOVES = ['F', 'L', 'R']  # Forward, Left, Right

    def update_position(self, pos):
        """Update current position and detect if robot is stuck."""
        now = time.time()
        if self.last_pos is None:
            self.last_pos = pos
            self.last_check_time = now
            return False  # Can't be stuck on first measurement

        if now - self.last_check_time >= self.CHECK_INTERVAL:
            dist_moved = np.linalg.norm(pos - self.last_pos)
            if dist_moved < self.MIN_MOVE_DIST:
                self.stuck_counter += 1
                print(f"[Mobility] Potentially stuck: moved {dist_moved:.3f} m, count {self.stuck_counter}")
                if self.stuck_counter >= self.STUCK_LIMIT:
                    self.stuck_counter = 0
                    print("[Mobility] Confirmed stuck! Initiating random 6-step sequence.")
                    self.random_sequence()
                    self.last_pos = pos
                    self.last_check_time = now
                    return True
            else:
                self.stuck_counter = 0
                print(f"[Mobility] Movement OK: moved {dist_moved:.3f} m")
            self.last_pos = pos
            self.last_check_time = now
        return False

    def random_sequence(self):
        """Execute a random 6-step sequence of movements."""
        sequence = [random.choice(self.POSSIBLE_MOVES) for _ in range(6)]
        print(f"[Mobility] Random sequence: {sequence}")
        for move in sequence:
            start_time = self.robot.getTime()
            duration = self.TURN_DURATION if move in ['L', 'R'] else self.FORWARD_DURATION
            while self.robot.getTime() - start_time < duration:
                if move == 'L':
                    self.leftMotor.setVelocity(-self.MAX_SPEED)
                    self.rightMotor.setVelocity(self.MAX_SPEED)
                elif move == 'R':
                    self.leftMotor.setVelocity(self.MAX_SPEED)
                    self.rightMotor.setVelocity(-self.MAX_SPEED)
                elif move == 'F':
                    self.leftMotor.setVelocity(self.MAX_SPEED)
                    self.rightMotor.setVelocity(self.MAX_SPEED)
                self.robot.step(self.timestep)
            self.leftMotor.setVelocity(0)
            self.rightMotor.setVelocity(0)
            self.robot.step(self.timestep)
        print("[Mobility] Random sequence completed.")