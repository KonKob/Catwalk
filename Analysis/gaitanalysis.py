import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import os
import imageio.v3 as iio
from scipy import interpolate
from scipy.signal import find_peaks, peak_widths
from statistics import mean
import cv2
import pickle


class Recording2D(ABC):
    
    def __init__(self, filepath: Path, recorded_framerate: int)->None:
        self.filepath = filepath
        self.full_df_from_hdf = self._get_df_from_hdf(filepath = filepath)
        self.recorded_framerate = recorded_framerate
        self.metadata = self._retrieve_metadata(filepath = filepath)
        
    def run(self, intrinsic_camera_calibration_filepath: Optional[Path]=None, video_filepath: Optional[Path] = None)->None:
        self._calculate_center_of_gravity()
        if intrinsic_camera_calibration_filepath != None and video_filepath != None:
            K, D = self._load_intrinsic_camera_calibration(intrinsic_camera_calibration_filepath = intrinsic_camera_calibration_filepath)
            image = iio.imread(video_filepath, index = 0)
            size = image.shape[1], image.shape[0]
            self.camera_parameters_for_undistortion = {'K': K, 'D': D, 'size': size}
        else:
            print("Distorted Points reflect the real-world distances poorly.\n To undistort, pass an intrinsic_camera_calibration_filepath and the video_filepath!")
            self.camera_parameters_for_undistortion = None
        self._create_all_bodyparts()
        self._normalize_coordinate_system()
        self._run_basic_operations_on_bodyparts()
        
    def run_gait_analysis(self)->None:
        """
        Function, that runs functions, necessary for gait analysis.

        Angles between bodyparts of interest are calculated as Angle objects.
        A peak detection algorithm on paw_speed is used to detect steps.
        EventBouts are created.
        """
        self._calculate_angles()
        self._detect_steps()
        self._calculate_parameters_for_gait_analysis()
        self.parameter_dict = { 'HindPawRight': 
                               {'angle_paw_knee_centerofgravity_right': self.angle_paw_knee_centerofgravity_right.parameter_array, 
                                'angle_paw_knee_bodyaxis_right': self.angle_paw_knee_bodyaxis_right.parameter_array, 
                                'angle_paw_secondfinger_bodyaxis_hind_right': self.angle_paw_secondfinger_bodyaxis_hind_right.parameter_array, 
                                'angle_paw_fifthfinger_bodyaxis_hind_right': self.angle_paw_fifthfinger_bodyaxis_hind_right.parameter_array, 
                                'hind_stance_right': self.hind_stance_right.parameter_array, 
                                'hind_stance': self.hind_stance, 
                                'area_hindpawright': self.area_hindpawright},
         'HindPawLeft': {'angle_paw_knee_centerofgravity_left': self.angle_paw_knee_centerofgravity_left.parameter_array,
                         'angle_paw_knee_bodyaxis_left': self.angle_paw_knee_bodyaxis_left.parameter_array, 
                         'angle_paw_secondfinger_bodyaxis_hind_left': self.angle_paw_secondfinger_bodyaxis_hind_left.parameter_array, 
                         'angle_paw_fifthfinger_bodyaxis_hind_left': self.angle_paw_fifthfinger_bodyaxis_hind_left.parameter_array, 
                         'hind_stance_left': self.hind_stance_left.parameter_array, 
                         'hind_stance': self.hind_stance, 
                         'area_hindpawleft': self.area_hindpawleft},
         'ForePawRight': {'angle_paw_secondfinger_bodyaxis_fore_left': self.angle_paw_secondfinger_bodyaxis_fore_left.parameter_array, 
                          'angle_paw_fifthfinger_bodyaxis_fore_left': self.angle_paw_fifthfinger_bodyaxis_fore_left.parameter_array, 
                          'fore_stance_left': self.fore_stance_left.parameter_array, 
                          'fore_stance': self.fore_stance},
         'ForePawLeft': {'angle_paw_secondfinger_bodyaxis_fore_right': self.angle_paw_secondfinger_bodyaxis_fore_right.parameter_array, 
                         'angle_paw_fifthfinger_bodyaxis_fore_right': self.angle_paw_fifthfinger_bodyaxis_fore_right.parameter_array, 
                         'fore_stance_right': self.fore_stance_right.parameter_array, 
                         'fore_stance': self.fore_stance}
        }
        self._add_angles_to_steps()
        self._create_PSTHs()
        self._parameters_when_paw_placed()
        
        
    def _get_df_from_hdf(self, filepath: Path)->pd.DataFrame:
        #if not filepath.endswith('.h5'):
        #   raise ValueError('The Path you specified is not linking to a .h5-file!')
        if not filepath.endswith('.csv'):
             raise ValueError('The Path you specified is not linking to a .csv-file!')
        df = pd.read_csv(filepath)
        #df = pd.read_hdf(filepath)
        df = df.drop('scorer', axis=1)
        df.columns = df.iloc[0, :]+ '_' + df.iloc[1, :]
        df = df.drop([0, 1], axis=0)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        df = df.astype(float)
        return df
    
    def _retrieve_metadata(self, filepath: str)->Dict:
        """
        relying on this file naming: 196_F7-27_220826_OTT_Bottom_synchronizedDLC_resnet152_OT_BottomCam_finalSep20shuffle1_550000filtered.h5
        """
        filepath_slices = filepath.split('_')
        animal = filepath_slices[2]
        paradigm = filepath_slices[3]
        recording_date = filepath_slices[4][:-3]
        return {'recording_date': recording_date, 'animal': animal, 'paradigm': paradigm}

    def _calculate_center_of_gravity(self)->None:
        for coordinate in ['x', 'y']:
            self.full_df_from_hdf[f'centerofgravity_{coordinate}'] = (self.full_df_from_hdf[f'Snout_{coordinate}'] + self.full_df_from_hdf[f'TailBase_{coordinate}'])/2
        self.full_df_from_hdf['centerofgravity_likelihood'] = self.full_df_from_hdf['Snout_likelihood']*self.full_df_from_hdf['TailBase_likelihood']
            
        
    def _create_all_bodyparts(self)->None:
        self.bodyparts = {}
        for key in self.full_df_from_hdf.keys():
            bodypart = key.split('_')[0]
            if bodypart not in self.bodyparts.keys() and bodypart != 'CenterOfGravity':
                self.bodyparts[bodypart] = Bodypart2D(bodypart_id = bodypart, df = self.full_df_from_hdf, camera_parameters_for_undistortion=self.camera_parameters_for_undistortion)
                
    
    def _normalize_coordinate_system(self)->None:
        pickle_filepath = Path(str(Path(self.filepath).parent.joinpath(self.metadata['animal'] + '_' + self.metadata['paradigm'] + '_' + self.metadata['recording_date'])) + ".pickle")
        if pickle_filepath.exists():      
            with open(pickle_filepath, "rb") as f:
                unpickler = pickle.Unpickler(f)
                mazecorners = unpickler.load()
        else:
            raise FileNotFoundError (f'Could not find a .pickle file for {self.filepath}!')
        
        x_offset, y_offset = mazecorners["offset_x"], mazecorners['offset_y']
        translation_vector = -np.array([x_offset, y_offset])       
        conversion_factor = self._get_conversion_factor_px_to_cm(length = int(mazecorners["length"]))
        rotation_angle = float(mazecorners["theta"])
        
        for bodypart in self.bodyparts.values():
            bodypart.normalize_df(translation_vector = translation_vector, rotation_angle = rotation_angle, conversion_factor = conversion_factor)
    
    def _get_conversion_factor_px_to_cm(self, length: float)->float:
        conversion_factor = (50/length)
        return conversion_factor
    
    def _load_intrinsic_camera_calibration(self, intrinsic_camera_calibration_filepath: Path) -> Tuple[np.array, np.array]:
        with open(intrinsic_camera_calibration_filepath, 'rb') as io:
            intrinsic_calibration = pickle.load(io)
        return intrinsic_calibration['K'], intrinsic_calibration['D']

        
    def _run_basic_operations_on_bodyparts(self)->None:
        for bodypart in self.bodyparts.values():
            bodypart.run_basic_operations(recorded_framerate = self.recorded_framerate)

        
    def _get_direction(self)->None:
        self.facing_towards_open_end = self._initialize_new_parameter(dtype=bool)
        self.facing_towards_open_end.loc[(self.bodyparts['Snout'].df.loc[:, 'x']>self.bodyparts['EarLeft'].df.loc[:, 'x']) &
                                    (self.bodyparts['Snout'].df.loc[:, 'x']>self.bodyparts['EarRight'].df.loc[:, 'x'])] = True
        
    def _get_turns(self)->None:
        turn_indices = self.facing_towards_open_end.where(self.facing_towards_open_end.diff()==True).dropna().index
        self.turns_to_closed=[EventBout2D(start_index = start_index) for start_index in turn_indices if self.facing_towards_open_end[start_index-1]==True]
        self.turns_to_open=[EventBout2D(start_index = start_index) for start_index in turn_indices if self.facing_towards_open_end[start_index-1]==False]
        for turning_bout in self.turns_to_closed:
            turning_bout.get_position(centerofgravity=self.bodyparts["centerofgravity"])
        for turning_bout in self.turns_to_open:
            turning_bout.get_position(centerofgravity=self.bodyparts["centerofgravity"])

    def _initialize_new_parameter(self, dtype: type)->pd.Series:
        """        
        pd.Series of an array in shape of n_frames with default values set to 0 (if dtype bool->False)
        """
        return pd.Series(np.zeros_like(np.arange(self.full_df_from_hdf.shape[0]), dtype = dtype))        
        
    def _detect_steps(self)->None:
        """
        Function that runs step detection in the individual paw Bodypart objects.
        """
        self.steps_per_paw = {paw : self.bodyparts[paw].detect_steps() for paw in ['HindPawRight', 'HindPawLeft', 'ForePawRight', 'ForePawLeft']}
        
    def _calculate_angles(self)->None:
        """
        creates angles objects that are interesting for gait analysis
        """
        self.angle_paw_knee_centerofgravity_left = Angle2D(bodypart_a = self.bodyparts['HindKneeleft'], bodypart_b = self.bodyparts['centerofgravity'], object_to_calculate_angle=self.bodyparts['HindPawLeft'])
        self.angle_paw_knee_centerofgravity_right = Angle2D(bodypart_a = self.bodyparts['HindKneeRight'], bodypart_b = self.bodyparts['centerofgravity'], object_to_calculate_angle=self.bodyparts['HindPawRight'])
        
        self.angle_paw_knee_bodyaxis_left = Angle2D(bodypart_a = self.bodyparts['HindPawLeft'], bodypart_b = self.bodyparts['HindKneeleft'], object_to_calculate_angle = (self.bodyparts['centerofgravity'], self.bodyparts['TailBase']))
        self.angle_paw_knee_bodyaxis_right = Angle2D(bodypart_a = self.bodyparts['HindPawRight'], bodypart_b = self.bodyparts['HindKneeRight'], object_to_calculate_angle = (self.bodyparts['centerofgravity'], self.bodyparts['TailBase']))
        
        self.angle_paw_secondfinger_bodyaxis_hind_left = Angle2D(bodypart_a = self.bodyparts['HindPawLeft'], bodypart_b = self.bodyparts['HindPawLeftSecondFinger'], object_to_calculate_angle = (self.bodyparts['centerofgravity'], self.bodyparts['TailBase']))
        self.angle_paw_fifthfinger_bodyaxis_hind_left = Angle2D(bodypart_a = self.bodyparts['HindPawLeft'], bodypart_b = self.bodyparts['HindPawLeftFifthFinger'], object_to_calculate_angle = (self.bodyparts['centerofgravity'], self.bodyparts['TailBase']))
        self.angle_paw_secondfinger_bodyaxis_hind_right = Angle2D(bodypart_a = self.bodyparts['HindPawRight'], bodypart_b = self.bodyparts['HindPawRightSecondFinger'], object_to_calculate_angle = (self.bodyparts['centerofgravity'], self.bodyparts['TailBase']))
        self.angle_paw_fifthfinger_bodyaxis_hind_right = Angle2D(bodypart_a = self.bodyparts['HindPawRight'], bodypart_b = self.bodyparts['HindPawRightFifthFinger'], object_to_calculate_angle = (self.bodyparts['centerofgravity'], self.bodyparts['TailBase']))
        
        self.angle_paw_secondfinger_bodyaxis_fore_left = Angle2D(bodypart_a = self.bodyparts['ForePawLeft'], bodypart_b = self.bodyparts['ForePawLeftSecondFinger'], object_to_calculate_angle = (self.bodyparts['Snout'], self.bodyparts['centerofgravity']))
        self.angle_paw_fifthfinger_bodyaxis_fore_left = Angle2D(bodypart_a = self.bodyparts['ForePawLeft'], bodypart_b = self.bodyparts['ForePawLegtFifthFinger'], object_to_calculate_angle = (self.bodyparts['Snout'], self.bodyparts['centerofgravity']))
        self.angle_paw_secondfinger_bodyaxis_fore_right = Angle2D(bodypart_a = self.bodyparts['ForePawRight'], bodypart_b = self.bodyparts['ForePawRightSecondFinger'], object_to_calculate_angle = (self.bodyparts['Snout'], self.bodyparts['centerofgravity']))
        self.angle_paw_fifthfinger_bodyaxis_fore_right = Angle2D(bodypart_a = self.bodyparts['ForePawRight'], bodypart_b = self.bodyparts['ForePawRightFifthFinger'], object_to_calculate_angle = (self.bodyparts['Snout'], self.bodyparts['centerofgravity']))
        
    
    def _add_angles_to_steps(self)->None:
        self.parameters_over_steps = {}
        for paw in ['HindPawRight', 'HindPawLeft', 'ForePawRight', 'ForePawLeft']:
            for parameter in self.parameter_dict[paw].keys():
                self.parameters_over_steps[parameter + '_' + paw] = np.array([self.parameter_dict[paw][parameter][step.start_index:step.end_index].values for step in self.bodyparts[paw].steps])
    
    def _calculate_parameters_for_gait_analysis(self)->None:
        self.hind_stance_right = Stance2D(paw=self.bodyparts['HindPawRight'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['TailBase'])
        self.hind_stance_left = Stance2D(paw=self.bodyparts['HindPawLeft'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['TailBase'])
        self.fore_stance_right = Stance2D(paw=self.bodyparts['ForePawRight'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['Snout'])
        self.fore_stance_left = Stance2D(paw=self.bodyparts['ForePawLeft'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['Snout'])
        
        self.hind_stance = self.hind_stance_right.parameter_array + self.hind_stance_left.parameter_array
        self.fore_stance = self.fore_stance_right.parameter_array + self.fore_stance_left.parameter_array
        
        self.area_hindpawright = abs(0.5 * (((self.bodyparts['HindPawRight'].df['x']-self.bodyparts['HindPawRightFifthFinger'].df['x'])*
        (self.bodyparts['HindPawRightSecondFinger'].df['y']-self.bodyparts['HindPawRightFifthFinger'].df['y']))-
        ((self.bodyparts['HindPawRightSecondFinger'].df['x']-self.bodyparts['HindPawRightFifthFinger'].df['x'])*
        (self.bodyparts['HindPawRight'].df['y']-self.bodyparts['HindPawRightFifthFinger'].df['y']))))
        
        self.area_hindpawleft = abs(0.5 * (((self.bodyparts['HindPawLeft'].df['x']-self.bodyparts['HindPawLeftFifthFinger'].df['x'])*
        (self.bodyparts['HindPawLeftSecondFinger'].df['y']-self.bodyparts['HindPawLeftFifthFinger'].df['y']))-
        ((self.bodyparts['HindPawLeftSecondFinger'].df['x']-self.bodyparts['HindPawLeftFifthFinger'].df['x'])*
        (self.bodyparts['HindPawLeft'].df['y']-self.bodyparts['HindPawLeftFifthFinger'].df['y']))))
    
    def _create_PSTHs(self)->None:
        self.parameters_as_psth = {}
        for parameter in self.parameters_over_steps:
            psth = self.parameters_over_steps[parameter].mean(axis = 0)
            self.parameters_as_psth[parameter] = psth
            
    def _parameters_when_paw_placed(self)->None:
        self.parameters_paw_placed = {}
        for parameter in self.parameters_over_steps:
            paw_placed = self.parameters_over_steps[parameter][:, 12].mean()
            self.parameters_paw_placed[parameter] = paw_placed
    
    
    
class Stance2D():
    """
    Class for calculation of the given paw to the bodyaxis as defined by object_a and object_b.

    Attributes:
        self.paw (Bodypart): Bodypart representation of the paw
        self.object_a: object, which defines the Bodyaxis
        self.object_b: object, which defines the Bodyaxis
        self.parameter_array: array, that contains the calculated distance (Stance) for every frame
    """
    def __init__(self, paw: 'Bodypart2D', object_a: 'Bodypart2D', object_b: 'Bodypart2D')->None:
        """
        Constructor for class Stance2D. It calls the functions to calculate the stance already.

        The Stance is stored in self.parameter_array

        Parameters:
            self.paw (Bodypart): Bodypart representation of the paw
            self.object_a: object, which defines the Bodyaxis
            self.object_b: object, which defines the Bodyaxis
        """
        self.paw = paw
        self.object_a = object_a
        self.object_b = object_b
        s = self._point_on_line_orthogonal_to_paw()
        self.parameter_array = self._calculate_distance(s=s)

    def _calculate_distance(self, s: Tuple[int, int])->float:
        """
        Function to calculate the distance between a point s and the paw.
        Returns:
            length(float): distance between s and the paw.
        """
        length = np.sqrt((self.paw.df['x']- s[0])**2 + 
                             (self.paw.df['y']-s[1])**2)
        return length


    def _point_on_line_orthogonal_to_paw(self)->Tuple[int, int]:
        """
        Function, that finds the point on the bodyaxis with the shortest distance to the given paw.

        First, the slope of the bodyaxis and the orthogonal on the bodyaxis is calculated as m1, m2. 
        Next, the intersection between the two lines and the y_axis is calculated as t1, t2.
        Last step, the coordinates of the intersection point sx and sy are calculated.

        Returns:
            Tuple: coordinates of the point on the bodyaxis with the shortest distance to the given paw.
        """
        #calculates the distance between the line given by a and b (intersection = s) and a point c
        m1 = (self.object_a.df['y'] - self.object_b.df['y']) / (self.object_a.df['x'] - self.object_b.df['x'])
        m2 = 1/(-m1)
        t1 = self.object_a.df['y'] - m1 * self.object_a.df['x']
        t2 = self.paw.df['y'] - m2 * self.paw.df['x']   
        sx = (t2 - t1)/(m1 - m2)
        sy = m2 * sx + t2
        return (sx, sy)
    

        
class Bodypart2D():
    
    def __init__(self, bodypart_id: str, df: pd.DataFrame, camera_parameters_for_undistortion: Optional[Dict]=None)->None:
        self.id = bodypart_id
        self._get_sliced_df(df = df)
        if camera_parameters_for_undistortion != None:
            self._undistort_points(camera_parameters_for_undistortion)
        else:
            self.df_points = self.df_raw
            
    
    def normalize_df(self, translation_vector: np.array, rotation_angle: float, conversion_factor: float)->None:
        translated_df = self._translate_df(translation_vector=translation_vector)
        rotated_df = self._rotate_df(rotation_angle=rotation_angle, df=translated_df)
        self.df = self._convert_df_to_cm(conversion_factor=conversion_factor, df=rotated_df)
        
    def run_basic_operations(self, recorded_framerate: int)->None:
        self._exclude_frames()
        self._get_speed(recorded_framerate = recorded_framerate)
        self._get_rolling_speed()
        
    def detect_steps(self)->List['Step']:
        speed = self.df["speed_cm_per_s"].copy()
        #Data smoothening:
        #x = np.arange(0, len(speed))
        #speed = np.nan_to_num(speed, copy=True)
        #spline = interpolate.UnivariateSpline(x, speed, s=1)
        #speed = spline(x)

        peaks = find_peaks(speed, prominence=50)
        steps_per_paw = self._create_steps(steps=peaks[0])
        self.steps = steps_per_paw
        return steps_per_paw
        
        
    def _get_sliced_df(self, df: pd.DataFrame)->None:
        self.df_raw = pd.DataFrame(data={'x': df.loc[:, self.id + '_x'], 'y': df.loc[:, self.id + '_y'], 'likelihood': df.loc[:, self.id + '_likelihood']})
    
    def _translate_df(self, translation_vector: np.array)->pd.DataFrame:
        translated_df = self.df_points.loc[:, ('x', 'y')] + translation_vector
        return translated_df
    
    def _rotate_df(self, rotation_angle: float, df: pd.DataFrame)->pd.DataFrame:
        cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
        rotated_df=pd.DataFrame()
        rotated_df['x'], rotated_df['y'] = df['x'] * cos_theta - df['y'] * sin_theta, df['x'] * sin_theta + df['y'] * cos_theta
        rotated_df['likelihood']=self.df_raw['likelihood']
        
        rotated_df.loc[:, ('y')] *= -1
        return rotated_df
    
    def _convert_df_to_cm(self, conversion_factor: float, df: pd.DataFrame)->pd.DataFrame:
        df.loc[:, ('x', 'y')]*=conversion_factor
        return df
        
    def _exclude_frames(self)->None:
        pass
    
    def _get_speed(self, recorded_framerate: int)->None:
        self.df.loc[:, 'speed_cm_per_s'] = np.NaN
        self.df.loc[:, 'speed_cm_per_s'] = (np.sqrt(self.df.loc[:, 'x'].diff()**2 + self.df.loc[:, 'y'].diff()**2)) / (1/recorded_framerate)        
    
    def _get_rolling_speed(self)->None:
        self.df.loc[:, 'rolling_speed_cm_per_s'] = np.NaN
        self.df.loc[:, 'rolling_speed_cm_per_s'] = self.df.loc[:, 'speed_cm_per_s'].rolling(5, min_periods=3, center=True).mean()
            
    def _create_steps(self, steps: List)->List['Step']:
        """
        Function, that creates Step objects for every speed peak inside of a gait event.
        """
        return [Step(paw = self.id, start_index = step_index) for step_index in steps if step_index > 7 and step_index < (self.df.shape[0] - 7)]
   
    def _undistort_points(self, camera_parameters_for_undistortion: Dict)->None:
        # understanding the maths behind it: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
        points = self.df_raw[['x', 'y']].copy().values
        new_K, _ = cv2.getOptimalNewCameraMatrix(camera_parameters_for_undistortion['K'], camera_parameters_for_undistortion['D'], camera_parameters_for_undistortion['size'], 1, camera_parameters_for_undistortion['size'])
        points_undistorted = cv2.undistortPoints(points, camera_parameters_for_undistortion['K'], camera_parameters_for_undistortion['D'], None, new_K)
        points_undistorted = np.squeeze(points_undistorted)
        self.df_points = pd.DataFrame()
        self.df_points[['x', 'y']] = points_undistorted
                
                    
                    
class EventBout2D():
    def __init__(self, start_index: int, end_index: Optional[int]=0)->None:
        self.start_index = start_index
        if end_index!=0:
            self.end_index = end_index
        else:
            self.end_index = start_index
        self._create_dict()

    def check_direction(self, facing_towards_open_end: pd.Series)->None:
        self.facing_towards_open_end = facing_towards_open_end.iloc[self.start_index]
        self.dict['facing_towards_open_end']=self.facing_towards_open_end

    def get_position(self, centerofgravity: Bodypart2D)->None:
        self.x_position=centerofgravity.df.loc[self.start_index, 'x']
        self.dict['x_position']=self.x_position

    def _create_dict(self)->None:
        self.dict = {}
        
        
        
class Angle2D():
    """
    depending on the input type of object_to_calculate_angle this class contains functions to calculate
    - the angle of 3 bodyparts to each other at the first given bodypart (bodypart_a) if object_to_calculate_angle is type Bodypart2D
    """
    def __init__(self, bodypart_a: Bodypart2D, bodypart_b: Bodypart2D, object_to_calculate_angle: Union[Bodypart2D, Tuple[Bodypart2D]])->None:
        self.bodypart_a = bodypart_a
        self.bodypart_b = bodypart_b
        if type(object_to_calculate_angle) == Bodypart2D:
            self.bodypart_c = object_to_calculate_angle
            self.parameter_array = self._calculate_angle_between_three_bodyparts()
        elif type(object_to_calculate_angle) == tuple:
            self.bodypart_c = object_to_calculate_angle[0]
            self.bodypart_d = object_to_calculate_angle[1]
            self.parameter_array = self._angle_between_two_lines()
        else:
            print('Could not calculate an angle for the given parameters')
        self.parameter_array = self.parameter_array.where(self.parameter_array>0, 180 + self.parameter_array)
        # converting all angles to positive values
        
    def _calculate_angle_between_three_bodyparts(self)->np.array:
        """
        calculates angle at bodypart_a
        """
        length_a = self._get_length_in_2d_space(self.bodypart_b, self.bodypart_c)
        length_b = self._get_length_in_2d_space(self.bodypart_a, self.bodypart_c)
        length_c = self._get_length_in_2d_space(self.bodypart_a, self.bodypart_b)
        return self._get_angle_from_law_of_cosines(length_a, length_b, length_c)
    
    def _get_length_in_2d_space(self, object_a: Bodypart2D, object_b: Bodypart2D) -> np.array:
        if hasattr(object_a, 'df'):       
            length = np.sqrt((object_a.df['x']-object_b.df['x'])**2 + 
                             (object_a.df['y']-object_b.df['y'])**2)
        else:
            length = np.sqrt((object_a.df_raw['x']-object_b.df_raw['x'])**2 + 
                             (object_a.df_raw['y']-object_b.df_raw['y'])**2)
        #theoretisch ist es nicht nötig, den normalisierten df zu nutzen, da der Winkel ja relativ bestimmt wird 
        #und sich daher im df zum df_raw nicht unterscheiden dürfte, dann müsste man allerdings für alle Rechnungen df_raw benutzen, 
        #damit auch die MazeCorners zum rotieren diese Klasse callen können
            
        return length
    
    def _get_angle_from_law_of_cosines(self, length_a: np.array, length_b: np.array, length_c: np.array)->np.array:
        cos_angle = (length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c)
        return np.degrees(np.arccos(cos_angle))
        
    
    def _angle_between_two_lines(self)->np.array:
        #calculates the angle at the intersection of two linear equations, each given by two points (a, b / c, d)
        #following the rule cos(angle)= (m1-m2)/(1+m1*m2)
        m1 = (self.bodypart_a.df['y'] - self.bodypart_b.df['y']) / (self.bodypart_a.df['x'] - self.bodypart_b.df['x'])
        m2 = (self.bodypart_c.df['y'] - self.bodypart_d.df['y']) / (self.bodypart_c.df['x'] - self.bodypart_d.df['x'])
        tan = (m1 - m2) / (1 + m1 * m2)
        angle = np.degrees(np.arctan(tan))
        return angle
        
        
        
class Step():
    def __init__(self, paw: str, start_index: int)->None:
        self.start_index = start_index - 7
        self.paw=paw
        self.end_index = start_index + 7
    