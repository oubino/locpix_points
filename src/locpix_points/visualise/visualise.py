# Take in datastructure and visualise as points or as images
import polars as pl
import open3d as o3d
import numpy as np
import matplotlib.colors as cl
import torch
import time

def visualise_pca(
    csv_path,
    x_name,
    y_name,
    z_name,
    channel_name,
    nn,
    cmap=["r", "darkorange", "b", "y"],
):
    """Read in .csv, calculate features for each point in the cloud and visualise

    Args:
        csv_loc (string): Location of the csv file
        x_name (string): Name of x column in csv
        y_name (string): Name of y column in csv
        z_name (string): Name of z column in csv
        nn (int): number of nearest neighbours to consider

    Returns:
        None
    """

    # Read in csv
    df = pl.read_csv(csv_path).select([x_name, y_name, z_name, channel_name])

    # List of pcds
    pcds = []

    def add_pcd(df, chan, cmap, pcds):
        """Function to take in dataframe, calculate pcd for channel chan and add to list of pcds

        Args:
            df (pandas dataframe): dataframe with the data
            chan (int): the channel to calculate the pcd for i.e. one pcd per channel in one colour
            cmap (list): list of strings each represents colour for channel
            pcds (list): List of pcds
        """

        pcd = o3d.geometry.PointCloud()
        coords = df.to_numpy()
        pcd.points = o3d.utility.Vector3dVector(coords)
        # pcd.paint_uniform_color(cl.to_rgb(cmap[chan]))
        pcds.append(pcd)

    # Get unique channels in dataframe
    channels = df.select(pl.col(channel_name).unique()).to_numpy()
    channels = sorted(channels)

    # Get rid of channel columns
    df = df.select([x_name, y_name, z_name])

    # Add pcd for each channel
    for index, channel in enumerate(channels):
        print("channel", channel)
        add_pcd(df, channel, cmap[index], pcds)

    # Extract features for point cloud
    output = extract_features(csv_path, x_name, y_name, z_name, nn)
    output = np.array(output)
    labels = np.array([1 if b[0] > b[1] else 0 for b in output], dtype="int32")

    # Alternative method
    # pcd.estimate_covariances(k=nn)
    # covs = np.asarray(pcd.covariances)
    # print(covs)

    print("linears", np.count_nonzero(labels == 1))
    print("non linears", np.count_nonzero(labels == 0))

    # colour point cloud based on labels
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # colors = colors[:,:3]
    colors = np.array([(1, 0, 0) if x == 1 else (0, 0, 1) for x in labels])
    for index, pcd in enumerate(pcds):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    class present:
        """This class is needed to define if a channel is currently being visualised"""

        def __init__(self):
            self.chan_zero_present = True
            self.chan_one_present = True
            self.chan_two_present = True
            self.chan_three_present = True

    present = present()

    def visualise_chan_0(vis):
        """Function used to add/remove channel during visualisation"""
        if present.chan_zero_present:
            vis.remove_geometry(pcds[3], False)
            present.chan_zero_present = False
        else:
            vis.add_geometry(pcds[3], False)
            present.chan_zero_present = True

    key_to_callback = {}
    key_to_callback[ord("K")] = visualise_chan_0
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)


def load_file(file, x_name, y_name, z_name, channel_name):
    if z_name is None:
        df = pl.read_parquet(file, columns=[x_name, y_name, channel_name])
    else:
        df = pl.read_parquet(file, columns=[x_name, y_name, z_name, channel_name])

    return df, df[channel_name].unique()


def add_pcd_parquet(df, chan, x_name, y_name, z_name, chan_name, unique_chans, cmap, pcds):
    if chan in unique_chans:
        pcd = o3d.geometry.PointCloud()

        if z_name is None:
            coords = (
                df.filter(pl.col(chan_name) == chan)
                .select([x_name, y_name])
                .to_numpy()
            )
            z = np.ones(coords.shape[0])
            z = np.expand_dims(z, axis=1)
            coords = np.concatenate([coords, z], axis=1)
        else:
            coords = (
                df.filter(pl.col(chan_name) == chan)
                .select([x_name, y_name, z_name])
                .to_numpy()
            )
        
        print(coords.shape)
        print(coords.dtype)
        print(type(coords))
        
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.paint_uniform_color(cl.to_rgb(cmap[chan]))
        pcds.append(pcd)
    return pcds


class Present:
    """Required for visualising the parquet file"""

    def __init__(self):
        self.chan_present = [True, True, True, True]

def visualise_parquet(
    file_loc,
    x_name,
    y_name,
    z_name,
    channel_name,
    channel_labels,
    cmap=["r", "darkorange", "b", "y"],
):
    """Visualise parquet file

    Args:
        file_loc (str) : Location of the parquet file to visualise
        x_name (str) : Name of the x column in the data
        y_name (str) : Name of the y column in the data
        z_name (str) : Name of the z column in the data, if is None, then
            assumes data is 2D
        channel_name (str) : Name of the channel column in the data
        channel_labels (dict) : Dictionary mapping channel label to name"""
    
    df, unique_chans = load_file(file_loc, x_name, y_name, z_name, channel_name)

    pcds = []

    cmap = ["r", "darkorange", "b", "y"]

    for key in channel_labels.keys():
        pcds = add_pcd_parquet(df, key, x_name, y_name, z_name, channel_name, unique_chans, cmap, pcds)
    
    visualise(pcds, unique_chans, channel_labels, cmap)

def visualise_torch_geometric(
        file_loc,
):
    """Visualise pytorch geometric object.
    Assumes that all locs are from the same channel and that there are also clusters present, plots these
    in two colours
    
    Args:
        file_loc (str) : Location of the pytorch geometric file"""
    
    x = torch.load(file_loc)

    cmap = ["r", "darkorange", "b", "y"]

    locs = x['locs'].pos.numpy()
    clusters = x['clusters'].pos.numpy()

    pcds = [locs, clusters]

    # convert 2d to 3d if required
    for index, pcd in enumerate(pcds):
        if pcd.shape[1] == 2:
            z = np.ones(pcd.shape[0])
            z = np.expand_dims(z, axis=1)
            pcds[index] = np.concatenate([pcd, z], axis=1)

    # loc to cluster edges
    lines = np.swapaxes(x['locs','in','cluster'].edge_index,0,1)
    # need one set of coordinates with all points in it 
    # add cluster coords at elocs nd of locs
    # increase index of cluster edge index by the number of locs
    coords = np.concatenate([pcds[0],pcds[1]],axis=0)
    lines[:,1] += len(locs)

    colors = [[0, 0, 1] for i in range(len(lines))]
    locs_to_clusters = o3d.geometry.LineSet()
    locs_to_clusters.points = o3d.utility.Vector3dVector(coords)
    locs_to_clusters.lines = o3d.utility.Vector2iVector(lines)
    locs_to_clusters.colors = o3d.utility.Vector3dVector(colors)

    # loc to loc edges
    lines = np.swapaxes(x['locs','clusteredwith','locs'].edge_index,0,1)
    lines = lines[10000:20000,:]
    colors = [[0, 1, 0] for i in range(len(lines))]
    locs_to_locs = o3d.geometry.LineSet()
    locs_to_locs.points = o3d.utility.Vector3dVector(pcds[0])
    locs_to_locs.lines = o3d.utility.Vector2iVector(lines)
    locs_to_locs.colors = o3d.utility.Vector3dVector(colors)

    # cluster to cluster edges
    lines = np.swapaxes(x['clusters','near','clusters'].edge_index,0,1)
    colors = [[1, 1, 0] for i in range(len(lines))]
    clusters_to_clusters = o3d.geometry.LineSet()
    clusters_to_clusters.points = o3d.utility.Vector3dVector(pcds[1])
    clusters_to_clusters.lines = o3d.utility.Vector2iVector(lines)
    clusters_to_clusters.colors = o3d.utility.Vector3dVector(colors)

    for index, pcd in enumerate(pcds):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        point_cloud.paint_uniform_color(cl.to_rgb(cmap[index]))
        pcds[index] = point_cloud
    
    visualise(
        pcds,
        locs_to_locs,
        locs_to_clusters,
        clusters_to_clusters,
        [0,1],
        {0:'locs', 1:'clusters'},
        cmap,

    )



def visualise(
    pcds,
    locs_to_locs,
    locs_to_clusters,
    clusters_to_clusters,
    unique_chans,
    channel_labels,
    cmap,
):
    """Visualise point cloud data

    Args:
        pcds (list) : List of point cloud data files
        locs_to_locs (list) : Lines to draw between localisations
        loc_to_clusters (list) : Lines to draw from localisations to clusters
        clusters_to_clusters (list) : Lines to draw between clusters
        unique_chans (list) : List of unique channels
        channel_labels (dict) : Dictionary mapping channel index to real name
        cmap (list) : Colours to plot in"""


    vis = o3d.visualization.Visualizer()
    

    assert len(pcds) == len(unique_chans)

    # pcd = df_to_feats(pcd, csv_path, 'X (nm)', 'Y (nm)', 'Z (nm)', 1000)

    present = Present()

    def visualise_chan_0(vis):
        if present.chan_present[0]:
            vis.remove_geometry(pcds[0], False)
            present.chan_present[0] = False
        else:
            vis.add_geometry(pcds[0], False)
            present.chan_present[0] = True

    def visualise_chan_1(vis):
        if present.chan_present[1]:
            vis.remove_geometry(pcds[1], False)
            present.chan_present[1] = False
        else:
            vis.add_geometry(pcds[1], False)
            present.chan_present[1] = True

    def visualise_chan_2(vis):
        if present.chan_present[2]:
            vis.remove_geometry(pcds[2], False)
            present.chan_present[2] = False
        else:
            vis.add_geometry(pcds[2], False)
            present.chan_present[2] = True

    def visualise_chan_3(vis):
        if present.chan_present[3]:
            vis.remove_geometry(pcds[3], False)
            present.chan_present[3] = False
        else:
            vis.add_geometry(pcds[3], False)
            present.chan_present[3] = True

    # reverse pcds for visualisation
    pcds.reverse()

    key_to_callback = {}
    key_to_callback[ord("K")] = visualise_chan_0
    key_to_callback[ord("R")] = visualise_chan_1
    key_to_callback[ord("T")] = visualise_chan_2
    key_to_callback[ord("Y")] = visualise_chan_3

    if 0 in channel_labels.keys():
        print(f"Channel 0 is {channel_labels[0]} is colour {cmap[0]} to remove use K")
    if 1 in channel_labels.keys():
        print(f"Channel 1 is {channel_labels[1]} is colour {cmap[1]} to remove use R")
    if 2 in channel_labels.keys():
        print(f"Channel 2 is ... is colour {cmap[2]} to remove use T")
    if 3 in channel_labels.keys():
        print(f"Channel 3 is ... is colour {cmap[3]} to remove use Y")

    pcds.append(locs_to_clusters)
    pcds.append(clusters_to_clusters)
    pcds.append(locs_to_locs)
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)
    
    """
    vis.create_window()
    vis.add_geometry(pcds[0])
    vis.add_geometry(pcds[1])
    # Alternatively, you can set the camera parameters directly
    # For example:
    view_control = vis.get_view_control()
    view_control.set_lookat([fov_x, fov_y, -2000])  # Set the look-at point
    #view_control.set_up([0, 1, 0])    # Set the up direction
    #view_control.set_front([0, 0, 1]) # Set the camera front direction       
    view_control.set_zoom(0.2)
    # v the renderer
    #vis.update_renderer()
    # Run the visualizer
    vis.run()
    vis.destroy_window()
    """

def main():
    visualise_torch_geometric(
        "tests/output/processed/train/0.pt"
    )
    visualise_parquet(
        "tests/output/preprocessed/featextract/locs/cancer_2.parquet", "x", "y", None, "channel", {0: "ereg"}
    )  # channels


if __name__ == "__main__":
    main()
