from typing import TypedDict

class Landmarks(TypedDict):
    right_eye: list[float, float]
    left_eye: list[float, float]
    nose: list[float, float]
    mouth_right: list[float, float]
    mouth_left: list[float, float]
    
class Face(TypedDict):
    score: float
    facial_area: list[int, int, int, int]
    landmarks: Landmarks