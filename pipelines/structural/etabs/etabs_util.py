import enum
import sys
from typing import Iterable

import comtypes.client

from pipelines.structural.etabs.unit import *


class MatType(enum.Enum):
    Steel = 1
    Concrete = 2
    NoDesign = 3
    Aluminum = 4
    ColdFormed = 5
    Rebar = 6
    Tendon = 7
    Masonry = 8


class Units_sys(enum.Enum):
    lb_in_F = 1
    lb_ft_F = 2
    kip_in_F = 3
    kip_ft_F = 4
    kN_mm_C = 5
    kN_m_C = 6
    kgf_mm_C = 7
    kgf_m_C = 8
    N_mm_C = 9
    N_m_C = 10
    Ton_mm_C = 11
    Ton_m_C = 12
    kN_cm_C = 13
    kgf_cm_C = 14
    N_cm_C = 15
    Ton_cm_C = 16


class ShellType(enum.Enum):
    ShellThin = 1
    ShellThick = 2
    Membrane = 3
    PlateThin_DO_NOT_USE = 4
    PlateThick_DO_NOT_USE = 5
    Layered = 6


class SlabType(enum.Enum):
    Slab = 0
    Drop = 1
    Stiff_DO_NOT_USE = 2
    Ribbed = 3
    Waffle = 4
    Mat = 5
    Footing = 6


def op_result(ret, op):
    if ret == 0:
        print(op, " successfully.")
    else:
        print(op, "failed.")


def handle_etabs_errors(func):
    def wrapper(*args, **kwargs):
        ret_value = func(*args, **kwargs)
        if ret_value != 0:
            print(f"ETABS operation {func.__name__} failed with return value: {ret_value}")
        else:
            print(f"ETABS operation {func.__name__} successfully.")
        return ret_value

    return wrapper


def create_ETABS_instance(attach_mode=False, program_path=""):
    def create_new():
        helper = comtypes.client.CreateObject("ETABSv1.Helper")
        helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)
        try:
            if program_path != "":
                myETABSObject = helper.CreateObject(program_path)
            else:
                myETABSObject = helper.CreateObjectProgID("CSI.ETABS.API.ETABSObject")
        except (OSError, comtypes.COMError):
            print("Cannot start a new instance of the program from " + program_path)
            op_result(-1, "failed to create ETABS API object")
            sys.exit(-1)

        myETABSObject.ApplicationStart()
        return myETABSObject

    if not attach_mode:
        myETABSObject = create_new()
        create_new_instance = True
    else:
        try:
            myETABSObject = comtypes.client.GetActiveObject("CSI.ETABS.API.ETABSObject")
            print("Attached to running instance of ETABS.")
            create_new_instance = False
        except (OSError, comtypes.COMError):
            print(
                "No running instance of the program found or failed to attach, try to create a new instance."
            )
            myETABSObject = create_new()
            create_new_instance = True
    SapModel = myETABSObject.SapModel
    op_result(0, "ETABS API initialized")

    return SapModel, myETABSObject, create_new_instance


@handle_etabs_errors
def define_Conc_Mat(etabs, material_name, E, miu, themal_exp, conc_level="C30"):
    names_ret = etabs.PropMaterial.GetNameList()
    if isinstance(names_ret, Iterable) and len(names_ret) >= 3:
        name_list = names_ret[1] if isinstance(names_ret[1], (list, tuple)) else []
        ret_code = names_ret[-1] if isinstance(names_ret[-1], int) else 0
        if ret_code == 0 and material_name in name_list:
            print(f"Material '{material_name}' already exists, skip redefining.")
            return 0

    ret = etabs.PropMaterial.AddMaterial(
        material_name, MatType.Concrete.value, "China", "GB", f"GB50010 {conc_level}"
    )
    ret = etabs.PropMaterial.SetMPIsotropic(material_name, E, miu, themal_exp)
    ret = etabs.PropMaterial.SetWeightAndMass(material_name, 2, 2550 * kg / M**3)
    return ret


@handle_etabs_errors
def define_Steel_Mat(etabs, material_name, E, miu, themal_exp, fy=345, fu=470, steel_level="Q345"):
    ret = etabs.PropMaterial.AddMaterial(material_name, MatType.Steel.value, "China", "GB", steel_level)
    ret = etabs.PropMaterial.SetMPIsotropic(material_name, E, miu, themal_exp)
    ret = etabs.PropMaterial.SetSteel(material_name, fy, fu)
    ret = etabs.PropMaterial.SetWeightAndMass(material_name, 2, 7890 * kg / M**3)
    return ret


@handle_etabs_errors
def define_wall_sec(etabs, wall_sec_name, shell_type, mat_name, thickness):
    ret = etabs.PropArea.SetWall(wall_sec_name, 1, shell_type, mat_name, thickness)
    return ret


@handle_etabs_errors
def define_slab_sec(etabs, floor_sec_name, slab_type, shell_type, mat_name, thickness):
    ret = etabs.PropArea.SetSlab(floor_sec_name, slab_type, shell_type, mat_name, thickness)
    return ret


@handle_etabs_errors
def define_beam_sec(etabs, beam_sec_name, mat_name, h, b):
    ret = etabs.PropFrame.SetRectangle(beam_sec_name, mat_name, h, b)
    return ret
