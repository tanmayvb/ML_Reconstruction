# load
volume = tiff.imread("volume.tiff")

df = pd.read_csv("filtered_points.csv")
points = df.iloc[:, -3:].values

# create mask
mask = create_gaussian_mask(volume.shape, points)

# save
tiff.imwrite("mask_gaussian.tiff", mask.astype(np.float32))
