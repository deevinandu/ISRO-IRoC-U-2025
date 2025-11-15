# main.py
import threading
import time
import config
from mavlink_manager import MavlinkManager
from algorithms import VisionSystem 
from navigation import NavigationController

def main():
    stop_event = threading.Event()
    vio_ready_event = threading.Event()
    mission_triggered = False
    RC_TRIGGER_THRESHOLD = 1900 

    mavlink_mgr = MavlinkManager(config.MAVLINK_DEVICE, config.MAVLINK_BAUD, config.COMPANION_COMPUTER_ID)
    vision_system = VisionSystem(mavlink_mgr, vio_ready_event, stop_event)
    nav_controller = NavigationController(mavlink_mgr, vision_system, vio_ready_event, stop_event)

    print("[MAIN] Connecting to vehicle...")
    mavlink_mgr.wait_for_heartbeat()
    
    vision_thread = threading.Thread(target=vision_system.run, name="VisionThread")
    nav_thread = threading.Thread(target=nav_controller.execute_mission, name="NAVThread")

    print("[MAIN] Starting Vision System thread...")
    vision_thread.start()
    
    print(f"\n[MAIN] Vision System is running. Flip RC Channel 10 (> {RC_TRIGGER_THRESHOLD} PWM) to start mission...")

    try:
        while not stop_event.is_set():
            mavlink_mgr.update()

            rc10_pwm = mavlink_mgr.rc_channels.get('chan10_raw', 0)
            if not mission_triggered and rc10_pwm > RC_TRIGGER_THRESHOLD:
                print(f"\n[MAIN] RC Trigger Detected! Starting navigation mission thread.")
                nav_thread.start()
                mission_triggered = True

            if not vision_thread.is_alive():
                print("[MAIN] CRITICAL: Vision System thread terminated. Shutting down.")
                stop_event.set()
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown requested by user (Ctrl+C).")
    finally:
        print("[MAIN] Sending stop signal to all threads...")
        stop_event.set()
        vision_thread.join()
        if mission_triggered:
            nav_thread.join()
        print("[MAIN] All threads have finished. Exiting.")

if __name__ == "__main__":
    main()
