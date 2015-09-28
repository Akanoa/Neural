import random
import pygame
import time
import math
import cPickle
import os.path
from docopt import docopt
from fparse import fparse

help = """Perceptron

Usage:
  perceptron.py train [--slow=<slow>] [--curve=<curve>] [--nb_points=<nb_points>] [--nb_trainings=<nb_training>] [--save_file=<save_file>]
  perceptron.py both [--slow=<slow>] [--curve=<curve>] [--nb_points=<nb_points>] [--nb_trainings=<nb_training>] [--save_file=<save_file>]
  perceptron.py exam [--slow=<slow>] [--curve=<curve>] [--nb_points=<nb_points>] [--save_file=<save_file>]

Options:
  -h --help                               Display this help.
  --slow=<slow>                           Slow down the animation rate [default: 0.01].
  --nb_points=<nb_points>                 Number of point use during training session [default: 1000].
  --nb_trainings=<nb_trainings>           Number of training before displaying final results [default: 3].
  --curve=<curve>                         Expression defining the training curve [default: x].
  --save_file=<save_file>                 Pickle file to save perceptron trainings.

Try to determine if a point is upper above a curve without know this curve :)
"""


class Perceptron(object):
    """
    The simplest neural net possible
    Takes 3 inputs, 2 numeric data and 1 bias
    And return a result following the sign of the of sum
    multiply by each weight input
    """

    def __init__(self, n=2, save_file=None):
        """
        Constructor initializes perceptron
        n = number of inputs excluding bias
        """
        # At first ways weights are initialized to random values
        self.weights = [round(random.uniform(-1.0, 1.0), 3) for weight in range(n + 1)]
        # Arbitrary chosen
        self.learning_control = 0.01
        # save_file used to dump or load perceptron state
        self.save_file = save_file

    def feeding(self, inputs):
        """
        Eats inputs and returns output
        inputs = a list of n values according to the number of inputs initializes
        return the output
        """
        processed_inputs = inputs[:]
        processed_inputs.append(1)
        inputs_sum = sum([input_value * self.weights[i] for i, input_value in enumerate(processed_inputs)])
        return (1, processed_inputs) if inputs_sum > 0 else (-1, processed_inputs)

    def train(self, inputs, desired):
        """
        For each input guess a answer and corrects all of its weights
        in case of error
        inputs = a list of n values according to the number of inputs initializes
        desired = an int representing th answer +1 good and -1 bad
        """
        guess, processed_inputs = self.feeding(inputs)
        error = desired - guess
        for i, weight in enumerate(self.weights):
            self.weights[i] += self.learning_control * error * processed_inputs[i];

    def exam(self, inputs, desired):
        """
        For each input guess a answer and corrects all of its weights
        in case of error
        inputs = a list of n values according to the number of inputs initializes
        desired = an int representing th answer +1 good and -1 bad
        """
        guess, processed_inputs = self.feeding(inputs)
        error = desired - guess
        return (inputs, 0, guess) if error != 0 else (inputs, 1, guess)

    def load(self):
        if self.save_file and os.path.isfile(self.save_file):
            with open(self.save_file, "rb") as fd:
                weights = cPickle.load(fd)
                self.weights = weights

    def save(self):
        if self.save_file:
            with open(self.save_file, "wb") as fd:
                cPickle.dump(self.weights, fd)

    def __repr__(self):
        val = "Weights: "
        for i, weight in enumerate(self.weights):
            val += " %s:%s"%(i, weight)
        val += " c=%s" % (self.learning_control)
        return val


class World:
    """
    Create a 2D environment to visualize Perceptron behaviors
    """

    def __init__(self, nb_points=10, dim=(800, 600), slower=0.1, training_function="100*sin(0.01*x)"):
        self.nb_points = int(nb_points)
        self.dim = dim
        self.points = []
        self.displayed_points = []
        self.previous_index = 0
        self.slower = float(slower)
        self.previous_time = time.time()
        self.training_function = fparse(training_function)
        pygame.init()

    def add_result(self, result):
        """
        Append a computed result to display
        """
        self.points.append(result)

    def shifting_point(self, point):
        return [point[0] + self.dim[0] / 2, -(point[1] - self.dim[1] / 2)]

    def slow_down(self, delta=0.1):
        """
        Append a point to the dispaying each delta second
        """
        # No more point can be displayed
        if self.previous_index == len(self.points):
            return

        current_time = time.time()
        currentDelta = current_time - self.previous_time

        if currentDelta > delta:
            self.previous_time = current_time
            self.displayed_points.append(self.points[self.previous_index])
            self.previous_index += 1

    def check_accurency(self):
        return round(100 * sum([point[1] for point in self.displayed_points]) / float(len(self.displayed_points)), 2)

    def final_accurency(self):
        return round(100 * sum([point[1] for point in self.points]) / float(len(self.points)), 2)

    def run(self):
        """
        Run the world
        """
        self.screen = pygame.display.set_mode(self.dim)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        ORANGE = (237, 195, 49)
        GREEN = (0, 255, 0)
        GREY = (212, 210, 210)
        font = pygame.font.Font(None, 36)

        final_accurency = self.final_accurency()
        final_accurency_text = font.render("Final Accuracy:  {0} %".format(final_accurency), 1, RED)
        end = font.render("END".format(final_accurency), 1, ORANGE)

        w, h = self.shifting_point(self.dim)
        line = [self.shifting_point([int(x), int(self.training_function(x=x))]) for x in range(-w / 2, w / 2)]

        # display loop
        run = True
        while run:
            self.screen.fill(WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # draw line
            pygame.draw.lines(self.screen, BLACK, False, line, 1)

            # slow down animation
            self.slow_down(self.slower)

            # write accurency
            accurency = self.check_accurency()
            accurency_text = font.render("Accuracy:  {0} %".format(accurency), 1, BLACK)
            # write number of points already displayed
            points_text = font.render("Points:  {0}/{1} ".format(len(self.displayed_points), len(self.points)), 1, BLACK)

            # display all points
            for point in self.displayed_points:
                coord, status, guess = point
                if guess > 0:
                    pygame.draw.circle(self.screen, BLUE, self.shifting_point(coord), 5, status)
                else:
                    pygame.draw.circle(self.screen, GREEN, self.shifting_point(coord), 5, status)

            frame = pygame.Surface((350, 170))
            frame.set_alpha(200)
            frame.fill(GREY)
            self.screen.blit(frame, (0, 0))
            self.screen.blit(final_accurency_text, (20, 20))
            self.screen.blit(accurency_text, (20, 60))
            self.screen.blit(points_text, (20, 100))
            if len(self.displayed_points) == len(self.points):
                self.screen.blit(end, (20, 140))
            pygame.display.flip()

    def generate_world(self):
        """
        Creates a 2D Cloud points world following dim
        return the world with the correct answer for each point
        """
        points = []
        for point in range(self.nb_points):
            point = [random.randrange(-self.dim[0] / 2, self.dim[0] / 2),
                     random.randrange(-self.dim[1] / 2, self.dim[1] / 2)]
            good = 1 if self.training_function(x=point[0]) > point[1] else -1
            points.append((point, good))
        return points


if __name__ == "__main__":
    arguments = docopt(help)
    p = Perceptron(2, save_file=arguments["--save_file"])
    world = World(nb_points=arguments["--nb_points"], slower=arguments["--slow"],
                  training_function=arguments["--curve"])
    values = world.generate_world()
    if arguments["train"] or arguments["both"]:
        p.load()
        print "training in progress..."
        for training in range(int(arguments["--nb_trainings"]) - 1):
            for point in values:
                p.train(*point)
        p.save()
        print "End of training"

    if arguments["exam"] or arguments["both"]:
        p.load()
        for i,point in enumerate(values):
            result = p.exam(*point)
            world.add_result(result)
        world.run()
