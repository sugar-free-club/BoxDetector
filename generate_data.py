import os
# import replicator envirnoment 
import omni.replicator.core as rep

# define the version of data
data_version = "5_closercam"

# setup random view range for camera: low point, high point
# widler range
# 4 version - sequential_pos = [(-1000, 220, -350),(1000, 220, 600)]
sequential_pos = [(-800, 220, -350),(800, 220, 500)]

# position of look-at target
look_at_position = (-212, 78, 57)


# setup working layer 
with rep.new_layer():

# define 3d models: usd format file source link, class, initial position  
    WORKSHOP = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd'
    CONVEYOR = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A23_PR_NVD_01.usd'
    CONVEYOR2 = 'E:/Workspace/sky/assets/Factory_Conveyer.usdz'
    BOX1     = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A3.usd'
    BOX2     = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_B3.usd'
    BOX3     = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_C3.usd'
    BOX4     = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_D3.usd'
    BOX5     = 'E:/Workspace/sky/assets/box/Laptop_box.usdz'
    BOX6     = 'E:/Workspace/sky/assets/box/Cardboard_Box_FREE__Agustin_Honnun.usdz'
    BOX8     = 'E:/Workspace/sky/assets/box/parcel_korea.usdz'
    BOX9     = 'E:/Workspace/sky/assets/box/carton.usdz'
    BOX10    = 'E:/Workspace/sky/assets/box/Cardboard_Box_Model_1Caja_Carton_Modelo_1.usdz'
    BOX11    = 'E:/Workspace/sky/assets/box/Caja_Girando.usdz'
    BOX12    = 'E:/Workspace/sky/assets/box/A534_Carton.usdz'
    BOX13    = 'E:/Workspace/sky/assets/box/CC0_-_Cardboard_Box_Closed.usdz'
    HUMAN1   = 'E:/Workspace/sky/assets/people/Nathan_Animated_003_-_Walking_3D_Man.usdz'
    HUMAN2   = 'E:/Workspace/sky/assets/people/Dennis_Posed_004_-_Male_Standing_Business_Model.usdz'
    HUMAN3   = 'E:/Workspace/sky/assets/people/Sophia_Animated_003_-_Animated_3D_Woman.usdz'
    CARGO1   = 'E:/Workspace/sky/assets/Pallet_Jack_Low_Poly.usdz'
    ARM1     = 'E:/Workspace/sky/assets/Smily_Arm.usdz'
    
    # generate the factory scene
    workshop = rep.create.from_usd(WORKSHOP)
    # generate 3 conveyors
    conveyor1 = rep.create.from_usd(CONVEYOR)
    conveyor2 = rep.create.from_usd(CONVEYOR)
    conveyor_customize = rep.create.from_usd(CONVEYOR2)
    # generate 15 box on the conveyors
    box1 = rep.create.from_usd(BOX1,semantics=[('class', 'box')])
    box2 = rep.create.from_usd(BOX2,semantics=[('class', 'box')])
    box3 = rep.create.from_usd(BOX3,semantics=[('class', 'box')])
    box4 = rep.create.from_usd(BOX4,semantics=[('class', 'box')])
    box5 = rep.create.from_usd(BOX4,semantics=[('class', 'box')])
    box6 = rep.create.from_usd(BOX2,semantics=[('class', 'box')])
    box_customize_1 = rep.create.from_usd(BOX5,semantics=[('class', 'box')])
    box_customize_2 = rep.create.from_usd(BOX6,semantics=[('class', 'box')])
    box_customize_4 = rep.create.from_usd(BOX8,semantics=[('class', 'box')])
    box_customize_5 = rep.create.from_usd(BOX9,semantics=[('class', 'box')])
    box_customize_6 = rep.create.from_usd(BOX10,semantics=[('class', 'box')])
    box_customize_7 = rep.create.from_usd(BOX11,semantics=[('class', 'box')])
    box_customize_8 = rep.create.from_usd(BOX12,semantics=[('class', 'box')])
    box_customize_9 = rep.create.from_usd(BOX13,semantics=[('class', 'box')])
    box_customize_10 = rep.create.from_usd(BOX8,semantics=[('class', 'box')])
    # generate people, cargo and robot arm.
    human1 = rep.create.from_usd(HUMAN1)
    human2 = rep.create.from_usd(HUMAN2)
    human3 = rep.create.from_usd(HUMAN3)
    cargo1 = rep.create.from_usd(CARGO1)
    arm1   = rep.create.from_usd(ARM1)
    
    # config the position of every asset
    #0
    with workshop:
        rep.modify.pose(
            position=(0,0,0),
            rotation=(0,-90,-90)
            )
    #1
    with conveyor1:
        rep.modify.pose(
            position=(-40,0,0),
            rotation=(0,-90,-90)
            )
    #2
    with conveyor2:
        rep.modify.pose(
            position=(-40,0,100),
            rotation=(-90,90,0)
            )
    #3
    with conveyor_customize:
        rep.modify.pose(
            position=(448,67,248),
            rotation=(0,-90,0),
            scale=(0.5,0.4,0.6)
            )
    #4
    with box1:
        rep.modify.pose(
            position=(-350,78,57),
            rotation=(0,-90,-90),
            scale=rep.distribution.uniform(1,1)
            )
    #5
    with box2:
        rep.modify.pose(
            position=(-100,78,57),
            rotation=(0,-90,-90),
            scale=rep.distribution.uniform(1,1)
            )
    #6
    with box3:
        rep.modify.pose(
            position=(100,78,57),
            rotation=(0,-90,-90),
            scale=rep.distribution.uniform(1,1)
            )
    #7
    with box4:
        rep.modify.pose(
            position=(200,70,57),
            rotation=(0,-90,-90),
            scale=rep.distribution.uniform(1,1)
            )
    #8
    with box5:
        rep.modify.pose(
            position=(15,78,50),
            rotation=(0,-90,-90),
            scale=rep.distribution.uniform(1,1)
            )
    #9
    with box6:
        rep.modify.pose(
            position=(300,78,57),
            rotation=(0,-90,-90),
            scale=rep.distribution.uniform(1,1)
            )
    #10
    with box_customize_1:
        rep.modify.pose(
            position=(170,82,-127),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.9,0.9)
            )
    #11
    with box_customize_2:
        rep.modify.pose(
            position=(-248,99,95),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.6,0.6)
            )
    #12
    with box_customize_4:
        rep.modify.pose(
            position=(459,106,97),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.1,0.1)
            ) 
    #13
    with box_customize_5:
        rep.modify.pose(
            position=(-253,90,240),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.1,0.1)
            ) 
    #14
    with box_customize_6:
        rep.modify.pose(
            position=(454,100,-78),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.6,0.6)
            )
    #15
    with box_customize_7:
        rep.modify.pose(
            position=(472,105,222),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.2,0.2)
            )
    #16
    with box_customize_8:
        rep.modify.pose(
            position=(455,104,-173),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.1,0.1)
            )
    #17
    with box_customize_9:
        rep.modify.pose(
            position=(461,100,340),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(1,1)
            )
    #18
    with box_customize_10:
        rep.modify.pose(
            position=(175,101,-15.9),
            rotation=(-90,0,0),
            scale=rep.distribution.uniform(0.1,0.1)
            ) 
    with human1:
        rep.modify.pose(
            position=(0,0,-120),
            rotation=(0,25,0),
            scale=rep.distribution.uniform(1,1)
        ) 
    with human2:
        rep.modify.pose(
            position=(565,0,60),
            rotation=(0,-110,0),
            scale=rep.distribution.uniform(0.6,0.6)
        ) 
    with human3:
        rep.modify.pose(
            position=(-400,0,200),
            rotation=(0,120,0),
            scale=rep.distribution.uniform(1,1)
        ) 
    with cargo1:
        rep.modify.pose(
            position=(173, 50, 173),
            rotation=(0,80,0),
            scale=rep.distribution.uniform(1,1)
        )
    with arm1:
        rep.modify.pose(
            position=(170,0,-320),
            rotation=(0,5,0),
            scale=rep.distribution.uniform(1,1)
        )
# define lighting function
    def sphere_lights(num):
    # generate light objects
        lights = rep.create.light(
                light_type="Sphere",
                temperature=rep.distribution.normal(3500, 500),
                intensity=rep.distribution.normal(15000, 5000),
                position=rep.distribution.uniform((-500, -300, -300), (500, 300, 300)),
                scale=rep.distribution.uniform(50, 100),
                color=rep.distribution.uniform((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                count=num
        )
        return lights
    rep.randomizer.register(sphere_lights)



# define function to create random position range for target  
    def get_shapes():
        shapes = rep.get.prims(semantics=[('class', 'box')])
        with shapes:
            rep.modify.pose(
                position=rep.distribution.uniform((0, -30, 0), (0, 30, 0)))#shape the range
        return shapes.node
    rep.randomizer.register(get_shapes)

# Setup camera and attach it to render product
    camera = rep.create.camera(position=sequential_pos[0], look_at=look_at_position)
    render_product = rep.create.render_product(camera, resolution=(512, 512))

    with rep.trigger.on_frame(num_frames=10000): #number of picture
        rep.randomizer.sphere_lights(6)    #number of lighting source 
        rep.randomizer.get_shapes()
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(sequential_pos[0],sequential_pos[1]), look_at=look_at_position)

# Initialize and attach writer for Kitti format data 
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(
            output_dir=os.path.join("E:/Workspace/sky/data/", data_version),
            bbox_height_threshold=25,
            fully_visible_threshold=0.95,
            omit_semantic_type=True
            )
    writer.attach([render_product])
    rep.orchestrator.preview()
