# track infe

def main_worker_tracking(**kwargs):
    from engine.parc_tracker import main_track_demo
    main_track_demo(**kwargs)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main_worker_tracking()
