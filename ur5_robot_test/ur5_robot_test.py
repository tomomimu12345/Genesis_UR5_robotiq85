"""
https://natsutan.hatenablog.com/entry/2025/01/29/012910

"""

import numpy as np  
import tkinter as tk  
import genesis as gs  
import threading  
import queue  
  
sliders = []  
pose_queue = queue.Queue()  
  
def update_pose(value) -> None:  
    pose = np.zeros([12], dtype=np.float32)  
    for i, slider in enumerate(sliders):  
        pose[i] = slider.get()  
  
    # pose をqueueに入れる  
    pose_queue.put(pose)  
  
def start_tkinter():  
    root = tk.Tk()  
    root.title('UR5 Control')  
    # ウィンドウの横幅を指定  
    root.geometry('250x600')  
  
    # 最初の3つはx, y, z  
    label1 = tk.Label(root, text='x, y, z')  
    label1.pack()  
  
    for i in range(3):  
        slider = tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=200)  
        slider.pack()  
        sliders.append(slider)  
  
    # 次の3つはrx, ry, rz  
    label2 = tk.Label(root, text='rx, ry, rz')  
    label2.pack()  
  
    for i in range(3):  
        slider = tk.Scale(root, from_=-3.14 / 2, to=3.14 / 2, resolution=0.01, orient=tk.HORIZONTAL, length=200)  
        slider.pack()  
        sliders.append(slider)  
  
    label3 = tk.Label(root, text='joints')  
    label3.pack()  
  
    for i in range(6):  
        slider = tk.Scale(root, from_=-3.14 , to=3.14, resolution=0.01, orient=tk.HORIZONTAL, length=200)  
        slider.pack()  
        sliders.append(slider)  
  
    # sliderの値が変わるとupdate_poseが呼ばれるようにする  
    for slider in sliders:  
        slider.config(command=update_pose)  
  
  
    root.mainloop()  
  
def main():  
    gs.init(backend=gs.gpu)  
  
    scene = gs.Scene(show_viewer=True)  
    plane = scene.add_entity(gs.morphs.Plane())  
    ur5 = scene.add_entity(  
        # ここにURDFファイルのパスを指定  
        gs.morphs.URDF(file='./example-robot-data/robots/ur_description/urdf/ur5_robot.urdf'),  
    )  
  
    jnt_names = [  
        'shoulder_pan_joint',  
        'shoulder_lift_joint',  
        'elbow_joint',  
        'wrist_1_joint',  
        'wrist_2_joint',  
        'wrist_3_joint',  
        'ee_fixed_joint'  
        ]  
  
  
    dofs_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  
  
    scene.build()  
  
    # sliderの表示を別スレッドで実行する  
    t = threading.Thread(target=start_tkinter)  
    t.start()  
  
    pose = np.zeros([12], dtype=np.float32)  
  
    ur5.set_dofs_position(pose, dofs_idx)  
  
    for i in range(1000):  
        scene.step()  
  
        # queueからposeを取り出す  
        pose = pose_queue.get()  
        print('pose:', pose)  
        ur5.set_dofs_position(pose, dofs_idx)  
  
  
# main  
if __name__ == '__main__':  
    main()