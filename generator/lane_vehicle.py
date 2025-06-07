import numpy as np
from . import BaseGenerator
from world import world_cityflow, world_cbengine # world_sumo #, world_openengine


class LaneVehicleGenerator(BaseGenerator):
    '''
    Generate state or reward based on statistics of lane vehicles.

    :param world: World object
    :param I: Intersection object
    :param fns: list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay", "lane_pressure" and "pressure". 
        "lane_count": get number of running vehicles on each lane. 
        "lane_waiting_count": get number of waiting vehicles(speed less than 0.1m/s) on each lane. 
        "lane_waiting_time_count": get the sum of waiting time of vehicles on the lane since their last action. 
        "lane_delay": the delay of each lane: 1 - lane_avg_speed/speed_limit.
        "lane_pressure": the number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        "pressure": difference of vehicle density between the in-coming lane and the out-going lane.

    :param in_only: boolean, whether to compute incoming lanes only. 
    :param average: None or str, None means no averaging, 
        "road" means take average of lanes on each road, 
        "all" means take average of all lanes.
    :param negative: boolean, whether return negative values (mostly for Reward).
    '''
    def __init__(self, world, I, fns, in_only=False, average=None, negative=False):
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads

        if isinstance(world, world_cityflow.World):
            for road in roads:
                from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
                self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
                
        elif isinstance(world, world_cbengine.World):
            for road in roads:
                from_zero = (road["start_inter"] == I.id) \
                    if self.world.RIGHT else (road["start_inter"] == I.id)
                self.lanes.append([int(road["id"])*100 + int(i) for i in range(road["num_lanes"])[::(1 if from_zero else -1)]])
        else:
            raise Exception('NOT IMPLEMENTED YET')

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = sum(len(x) for x in self.lanes)
        if average == "road":
            size = len(roads)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size
        # if self.ob_length == 3:
        #     self.ob_length = 4

        self.average = average
        self.negative = negative

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]

        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]
            if self.I.id in result and self.fns == ['pressure']:
                ret = np.append(ret, result[self.I.id])
                continue
            
            fn_result = np.array([])
            for road_lanes in self.lanes:
                road_result = []
                for lane_id in road_lanes:
                    if lane_id in result.keys():
                        road_result.append(result[lane_id])
                    else:
                        road_result.append(0)
                if self.average == "road" or self.average == "all":
                    road_result = np.mean(road_result)
                else:
                    road_result = np.array(road_result)
                fn_result = np.append(fn_result, road_result)
            
            if self.average == "all":
                fn_result = np.mean(fn_result)
            ret = np.append(ret, fn_result)
        if self.negative:
            ret = ret * (-1)
        return ret

if __name__ == "__main__":
    from world.world_cityflow import World
    world = World("examples/configs.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())

