import numpy as np
from . import BaseGenerator

class LaneVehicleGeneratorV2(BaseGenerator):
    def __init__(self, world, I, fns=['lane_count', 'lane_waiting_count'], targets=['inlane_2_num_vehicle','outlane_2_num_vehicle','inlane_2_num_waiting_vehicle']):
        self.world = world
        self.I = I
        self.world.subscribe(fns)
        self.fns = fns
        self.targets = targets
        
        self.in_lanes = self.I.in_lanes
        self.out_lanes = self.I.out_lanes
        self.ob_length = len(self.in_lanes)
        
        self.result_functions = {
            "inlane_2_num_vehicle": self.inlane_2_num_vehicle,
            "outlane_2_num_vehicle": self.outlane_2_num_vehicle,
            "inlane_2_num_waiting_vehicle": self.inlane_2_num_waiting_vehicle,
        }
        
    def inlane_2_num_vehicle(self, fns):
        return np.array([fns['lane_count'].get(lane, 0) for lane in self.in_lanes])
    
    def outlane_2_num_vehicle(self, fns):
        return np.array([fns['lane_count'].get(lane, 0) for lane in self.out_lanes])
    
    def inlane_2_num_waiting_vehicle(self, fns):
        return np.array([fns['lane_waiting_count'].get(lane, 0) for lane in self.in_lanes])

    def generate(self):
        fns = {fn:self.world.get_info(fn) for fn in self.fns}
        ret = [self.result_functions[res](fns) for res in self.targets]
        return ret

if __name__ == "__main__":
    from world.world_cityflow import World
    world = World("examples/configs.json", thread_num=1)
    laneVehicle = LaneVehicleGeneratorV2(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())

