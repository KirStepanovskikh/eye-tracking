import numpy as np
import pandas as pd

from bottleneck import move_mean
from typing import Tuple


class IVT:
    """Velocity Threshold Algorithm for Fixation Identification"""
    def __init__(self,
                 dist_to_screen: float,
                 resolution: Tuple[int, int],
                 diagonal_inch: float) -> None:
        """
        Initialize subject's experiment parameters
        :param dist_to_screen: distance from the participant's eyes to the screen in centimetres
        :param resolution: width and height of screen in pixels
        :param diagonal_inch: diagonal screen size in inches
        """
        self.dist_to_screen = dist_to_screen
        self.resolution = resolution
        self.diagonal_inch = diagonal_inch

        # initialize attributes
        self.x_smoothed = np.nan
        self.y_smoothed = np.nan
        self.euclidean_dist = np.nan
        self.visual_angle = np.nan
        self.time = np.nan
        self.velocity = np.nan
        self.summary = np.nan

    def get_euclidean_dist(self,
                           x_coord: np.array,
                           y_coord: np.array) -> None:
        """
        Calculate the distance between the current and next gaze position
        :param x_coord: vector of x coordinates in pixels
        :param y_coord: vector of y coordinates in pixels
        """
        x_diff = np.diff(x_coord, prepend=np.nan)
        y_diff = np.diff(y_coord, prepend=np.nan)
        self.euclidean_dist = np.sqrt(x_diff ** 2 + y_diff ** 2)

    def get_visual_angle(self,
                         dist_on_screen: np.array) -> None:
        """
        Convert distances on the screen in pixels to visual angles in degrees
        :param dist_on_screen: vector of distances on the screen in pixels
        """
        # calculate ppi (pixels per inch)
        diagonal_pxl = (self.resolution[0] ** 2 + self.resolution[1] ** 2) ** 0.5
        ppi = diagonal_pxl / self.diagonal_inch
        # convert distances to cm
        dist_on_screen = dist_on_screen / ppi * 2.54

        radians = 2 * np.arctan(0.5 * dist_on_screen / self.dist_to_screen)
        self.visual_angle = radians * 180 / np.pi

    def get_velocity(self,
                     visual_angle: np.array,
                     time: np.array) -> None:
        """
        Calculate velocity in degrees per second of gaze motions
        :param visual_angle: vector of visual angles in degrees
        :param time: vector of gaze motion timestamps in ms
        """
        self.time = time
        elapsed_time = np.diff(time, prepend=np.nan) / 1000
        self.velocity = visual_angle / elapsed_time

    def _get_fixation_clusters(self,
                               velocity_threshold: float,
                               duration_threshold: float) -> pd.DataFrame:
        """
        Extract gaze fixation groups and calculate summaries
        :param velocity_threshold: velocity in degrees per second specifies gaze points below as fixations
        :param duration_threshold: time in ms specifies gaze points above as fixations
        :return: summaries of gaze fixation groups
        """
        is_motion = self.velocity > velocity_threshold
        last_sample_is_motion = False

        # initialize arrays of fixation samples
        n_cluster = 1
        cluster_time = np.array([])
        x_coord = np.array([])
        y_coord = np.array([])
        cluster_velocity = np.array([])

        # initialize fixation clusters summary arrays
        clusters = np.array([])
        cluster_time_from = np.array([])
        cluster_time_to = np.array([])
        cluster_duration = np.array([])
        cluster_x_centroid = np.array([])
        cluster_y_centroid = np.array([])
        cluster_variance = np.array([])
        cluster_average_velocity = np.array([])

        samples = range(self.velocity.size)
        for n_sample in samples:
            # get sample data
            sample_time = self.time[n_sample]
            sample_x = self.x_smoothed[n_sample]
            sample_y = self.y_smoothed[n_sample]
            sample_velocity = self.velocity[n_sample]
            current_sample_is_motion = is_motion[n_sample]

            # fixation cluster is ended
            if not current_sample_is_motion and last_sample_is_motion:
                t1 = min(cluster_time)
                t2 = max(cluster_time)
                duration = t2 - t1
                # save summary of fixation cluster
                if duration > duration_threshold:
                    # calculate summary of current cluster
                    x_centroid = np.mean(x_coord)
                    y_centroid = np.mean(y_coord)
                    variance = np.sqrt(np.std(x_coord) ** 2 + np.std(y_coord) ** 2)
                    average_velocity = np.mean(cluster_velocity)

                    # save summary of fixation current cluster to summary arrays
                    clusters = np.append(clusters, n_cluster)
                    cluster_time_from = np.append(cluster_time_from, t1)
                    cluster_time_to = np.append(cluster_time_to, t2)
                    cluster_duration = np.append(cluster_duration, duration)
                    cluster_x_centroid = np.append(cluster_x_centroid, x_centroid)
                    cluster_y_centroid = np.append(cluster_y_centroid, y_centroid)
                    cluster_variance = np.append(cluster_variance, variance)
                    cluster_average_velocity = np.append(cluster_average_velocity, average_velocity)
                    # update cluster number
                    n_cluster += 1

                # start new fixation cluster
                cluster_time = np.array([])
                x_coord = np.array([])
                y_coord = np.array([])
                cluster_velocity = np.array([])

            # add fixation sample to current fixation cluster
            if not current_sample_is_motion:
                # replace NaN to 0
                sample_velocity = 0 if sample_velocity != sample_velocity else sample_velocity
                sample_x = 0 if sample_x != sample_x else sample_x
                sample_y = 0 if sample_y != sample_y else sample_y

                cluster_time = np.append(cluster_time, sample_time)
                x_coord = np.append(x_coord, sample_x)
                y_coord = np.append(y_coord, sample_y)
                cluster_velocity = np.append(cluster_velocity, sample_velocity)

            # move to next sample
            last_sample_is_motion = current_sample_is_motion

        # the last fixation cluster is ended
        if cluster_time.size > 0:
            t1 = min(cluster_time)
            t2 = max(cluster_time)
            duration = t2 - t1
            # save summary of fixation cluster
            if duration > duration_threshold:
                # calculate summary of current cluster
                x_centroid = np.mean(x_coord)
                y_centroid = np.mean(y_coord)
                variance = np.sqrt(np.std(x_coord) ** 2 + np.std(y_coord) ** 2)
                average_velocity = np.mean(cluster_velocity)

                # save summary of fixation current cluster to summary arrays
                clusters = np.append(clusters, n_cluster)
                cluster_time_from = np.append(cluster_time_from, t1)
                cluster_time_to = np.append(cluster_time_to, t2)
                cluster_duration = np.append(cluster_duration, duration)
                cluster_x_centroid = np.append(cluster_x_centroid, x_centroid)
                cluster_y_centroid = np.append(cluster_y_centroid, y_centroid)
                cluster_variance = np.append(cluster_variance, variance)
                cluster_average_velocity = np.append(cluster_average_velocity, average_velocity)

        # fixation clusters not found
        if clusters.size == 0:
            clusters = np.array([np.nan])

        fixation_clusters = pd.DataFrame({
            "fixation": clusters,
            "time_from": cluster_time_from,
            "time_to": cluster_time_to,
            "duration": cluster_duration,
            "x_centroid": cluster_x_centroid,
            "y_centroid": cluster_y_centroid,
            "variance": cluster_variance,
            "average_velocity": cluster_average_velocity
        })
        return fixation_clusters.set_index("fixation")

    def identify_fixation(self,
                          time: np.array,
                          x_coord: np.array,
                          y_coord: np.array,
                          velocity_threshold: float,
                          duration_threshold: float) -> None:
        """
        run velocity threshold algorithm
        :param time: vector of gaze motion timestamps in ms
        :param x_coord: vector of x coordinates in pixels
        :param y_coord: vector of y coordinates in pixels
        :param velocity_threshold: velocity in degrees per second specifies gaze points below as fixations
        :param duration_threshold: time in ms specifies gaze points above as fixations
        """
        # noise reduction using smoothing of moving average
        self.x_smoothed = move_mean(x_coord, window=5)
        self.y_smoothed = move_mean(y_coord, window=5)

        self.get_euclidean_dist(x_coord=self.x_smoothed,
                                y_coord=self.y_smoothed)

        self.get_visual_angle(dist_on_screen=self.euclidean_dist)

        self.get_velocity(visual_angle=self.visual_angle,
                          time=time)

        self.summary = self._get_fixation_clusters(velocity_threshold,
                                                   duration_threshold)
