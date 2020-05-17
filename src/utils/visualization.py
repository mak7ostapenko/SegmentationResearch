import os

import cv2
import numpy as np


def three_stage_model_plot_results(image, input_image_path, output_path, target_points, num_coord_dict):
    """"""
    input_image = image.copy()
    out_image_path = os.path.join(output_path, input_image_path.split('/')[-1])
    font = cv2.FONT_HERSHEY_SIMPLEX

    if num_coord_dict is not None:
        for number, coord in num_coord_dict.items():
            center = (int(coord[0]), int(coord[1]))
            cv2.putText(input_image, number, center, font, 1, (0, 0, 255))

    if target_points is not None:
        for target_point in target_points:
            center = (int(target_point[0]), int(target_point[1]))
            cv2.circle(input_image, center, 3, (0, 0, 255), -1)

    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_image_path, input_image)


def three_stage_model_plot_matching_results(input_image, input_image_path, output_path, num_target_point_dict):
    """

    :param input_image:
    :param input_image_path:
    :param output_path:
    :param num_target_point_dict: key = number to plot, value = coordinates of target point
    :return:
    """
    out_image_path = os.path.join(output_path, input_image_path.split('/')[-1])
    font = cv2.FONT_HERSHEY_SIMPLEX

    for number, coord in num_target_point_dict.items():
        center = (int(coord[0]), int(coord[1]))
        cv2.circle(input_image, center, 5, (0, 230, 0), -1)
        cv2.putText(input_image, number, coord, font, 2, (0, 0, 255))

        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_image_path, input_image)


def blade_distance_meas_visualization(image, dist_between_edges, predicted_dist,
                                      left_curve, right_curve, out_path, image_path):
    """
    Visualization of distance between a robot and a edge of blade.

    :param image: np.array, image on which will be plotted result.
    :param dist_between_edges: float, distance in pixels between two edges of the blade.
    :param predicted_dist: float, predicted distance between a robot and a edge of blade.
    :param left_curve: np.array with shape=(n_dots, 2), set of dots.
    :param right_curve: np.array with shape=(n_dots, 2), set of dots.
    :param out_path: str, path to output image.
    :param image_path: str, path to input image.
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_curve_x, left_curve_y = left_curve
    right_curve_x, right_curve_y = right_curve

    if dist_between_edges < 300:
        edge_color = (0, 0, 150)

    elif dist_between_edges >= 300 and dist_between_edges < 500:
        edge_color = (0, 0, 230)

    elif dist_between_edges >= 500 and dist_between_edges < 700:
        edge_color = (200, 0, 0)

    else:
        edge_color = (0, 200, 0)

    if (len(right_curve_x) == 0) or (len(right_curve_y) == 0) or \
            (len(left_curve_x) == 0) or (len(left_curve_x) == 0):
        return

    cv2.line(image, (right_curve_x[0], right_curve_y[0]), (right_curve_x[-1], right_curve_y[-1]), edge_color, 5)
    # plot distance line
    cv2.line(image, (int(np.min(left_curve_x)), right_curve_y[30]),
             (right_curve_x[30], right_curve_y[30]), (0, 255, 0), 5)

    # write on image dists
    cv2.putText(image, 'Predicted distance = {}'.format(predicted_dist), (70, 130), font, 1, edge_color, 2, cv2.LINE_AA)

    out_image_path = os.path.join(out_path, image_path.split('/')[-1])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_image_path, image)
