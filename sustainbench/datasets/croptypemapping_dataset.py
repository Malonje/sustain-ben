import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sustainbench.datasets.sustainbench_dataset import SustainBenchDataset
from sustainbench.common.grouper import CombinatorialGrouper
from sustainbench.common.utils import subsample_idxs, shuffle_arr
import os
import json

import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from constants import BANDS, NUM_CLASSES, GRID_SIZE, CM_LABELS, CROPS, MEANS, STDS

# BAND STATS



PLANET_DIM = 212

partition_to_idx = {
    "train": 0,
    "val": 1,
    "test": 2,
}


class CropTypeMappingDataset(SustainBenchDataset):
    """
    Supported `split_scheme`:
        'official' - same as 'ghana'
        'south-sudan'

    Input (x):
        List of three satellites, each containing C x 64 x 64 x T satellite image,
        with 12 channels from S2, 2 channels from S1, and 6 from Planet.
        Additional bands such as NDVI and GCVI are computed for Planet and S2.
        For S1, VH/VV is also computed. Time series are zero padded to 256.
        Mean/std applied on bands excluding NDVI and GCVI. Paper uses 32x32
        imagery but the public dataset/splits use 64x64 imagery, which is
        thusly provided. Bands are as follows:

        S1 - [VV, VH, RATIO]
        S2 - [BLUE, GREEN, RED, RDED1, RDED2, RDED3, NIR, RDED4, SWIR1, SWIR2, NDVI, GCVI]
        PLANET - [BLUE, GREEN, RED, NIR, NDVI, GCVI]

    Output (y):
        y is a 64x64 tensor with numbers for locations with a crop class.

    Metadata:
        Metadata contains integer in format {Year}{Month}{Day} for each image in
        respective time series. Zero padded to 256, can be used to derive a mask.

    Website: https://github.com/roserustowicz/crop-type-mapping

    Original publication:
    @InProceedings{Rustowicz_2019_CVPR_Workshops,
        author = {M Rustowicz, Rose and Cheong, Robin and Wang, Lijing and Ermon, Stefano and Burke, Marshall and Lobell, David},
        title = {Semantic Segmentation of Crop Type in Africa: A Novel Dataset and Analysis of Deep Learning Methods},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month = {June},
        year = {2019}
    }

    License:
        S1/S2 data is U.S. Public Domain.

    """
    _dataset_name = 'africa_crop_type_mapping'
    _versions_dict = {  # TODO
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official',
                 resize_planet=False, calculate_bands=True, normalize=True,
                 l8_bands=[0,1,2,3,4,5,6,7], s1_bands=[0,1,2], s2_bands=[0,1,2,3,4,5,6,7,8,9],
                 ps_bands=[0,1,2,3], truth_mask=10, img_dim=(32,32)):
        """
        Args:
            resize_planet: True if Planet imagery will be resized to 64x64
            calculate_bands: True if aditional bands (NDVI and GCVI) will be calculated on the fly and appended
            normalize: True if bands (excluding NDVI and GCVI) wll be normalized
        """
        self._resize_planet = resize_planet
        self._calculate_bands = calculate_bands
        self._normalize = normalize

        self.l8_bands = l8_bands
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.ps_bands = ps_bands
        self.truth_mask = truth_mask
        self.img_dim = img_dim


        if calculate_bands:
            self.l8_bands.extend([-2, -1])
            self.s1_bands = s1_bands
            self.s2_bands.extend([-2, -1])
            self.ps_bands.extend([-2, -1])
            # print(self.l8_bands)



        self._version = version
        if split_scheme == "cauvery":
            self._data_dir = root_dir
        else:
            self._data_dir = self.initialize_data_dir(root_dir, download)

        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme not in ['official', 'ghana', 'southsudan', 'cauvery']:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        if self._split_scheme in ['official', 'ghana']:
            self._country = 'ghana'
        if self._split_scheme in ['southsudan']:
            self._country = 'southsudan'
        if self._split_scheme in ['cauvery']:
            self._country = 'cauvery'

        if self._split_scheme in ['official', 'ghana', 'southsudan']:
            split_df = pd.read_csv(os.path.join(self.data_dir, self._country, 'list_eval_partition.csv'))
            ## made changes from here
            sa = split_df['partition'].values
            le = len(sa)
            gridid = [0] * le + [1] * le + [2] * le + [3] * le
            newd = pd.concat([split_df, split_df, split_df, split_df])
            newd['grid_id'] = gridid
            split_df = newd
            self.grid_id = split_df['grid_id'].values
        else:
            split_df = pd.read_csv(os.path.join(self.data_dir, self._country, 'cauvery_dataset.csv'))
            l = [1677, 1887, 1571, 1753, 2046, 1617, 1626, 1881, 1587, 1661, 1739, 2329, 2042, 1778, 1962, 2314, 1622, 2391, 2304, 1878, 2293, 1540, 2372, 1945, 2406, 2031, 1666, 1720, 2236, 2347, 1993, 1591, 1646, 1971, 1551, 2382, 1902, 2005, 1507, 2396, 1954, 1978, 2004, 1801, 1636, 2011, 2378, 1535, 1561, 2041, 2311, 1653, 2243, 2032, 2239, 1749, 1766, 1738, 1590, 1750, 2034, 1557, 1606, 1774, 1542, 1584, 1934, 1799, 1958, 1654, 1909, 1936, 1570, 1682, 1655, 1642, 1628, 1903, 2259, 1867, 1672, 1524, 1651, 2359, 2307, 2336, 1583, 2127, 1969, 2410, 1781, 2319, 1908, 2122, 1846, 1847, 2294, 2027, 1518, 1578, 1691, 1836, 1502, 1959, 1619, 1782, 2234, 1718, 1658, 2276, 1595, 1603, 1693, 1683, 2006, 2303, 1656, 1756, 1767, 2402, 2103, 1552, 1946, 1517, 1939, 2022, 2136, 1506, 1657, 1891, 1745, 1680, 1975, 1592, 1732, 2265, 1852, 1943, 1758, 1898, 2272, 2323, 1545, 1837, 2369, 1569, 2408, 1851, 2282, 1597, 2049, 2256, 1662, 2232, 1864, 1648, 1689, 1503, 1735, 1928, 2365, 1515, 1906, 1650, 1780, 1611, 2038, 1844, 2138, 2128, 1822, 1809, 1951, 1791, 1793, 2291, 1768, 1807, 2299, 1715, 1880, 1794, 1520, 2261, 2113, 1901, 1727, 2353, 2003, 1869, 1876, 2051, 1556, 2298, 1817, 2121, 2318, 1938, 1690, 1706, 2250, 1564, 2285, 2328, 1679, 1741, 2362, 1598, 1922, 1614, 2342, 1668, 1811, 1678, 1961, 1980, 2117, 2134, 2270, 1627, 1860, 2233, 1696, 1988, 1536, 1772, 2309, 1645, 1640, 1635, 2021, 2331, 1576, 2106, 2326, 2030, 1562, 2351, 1843, 1984, 1647, 1529, 1605, 2059, 1669, 1862, 2033, 1894, 2364, 1711, 2114, 1786, 1989, 2355, 1523, 1900, 1543, 1493, 1492, 2281, 2017, 2356, 1870, 1723, 1593, 1888, 2047, 2137, 1953, 1816, 2398, 1979, 2348, 1512, 2321, 2409, 2334, 2335, 1710, 2395, 1688, 2026, 1729, 2109, 2125, 1539, 1819, 2105, 2386, 2390, 2238, 1792, 2288, 1546, 1609, 1973, 1631, 1806, 2116, 2001, 1859, 1986, 1845, 1829, 2044, 2112, 1823, 1549, 1525, 2301, 2000, 1960, 1558, 1699, 1835, 2251, 1886, 1897, 2244, 2018, 1703, 1568, 1821, 2279, 2002, 1916, 2052, 1858, 2357, 2327, 2241, 1981, 2056, 1990, 1719, 2374, 2235, 2379, 2253, 1555, 2039, 2013, 1927, 1522, 1805, 1638, 2306, 2302, 1917, 1704, 1997, 1957, 2371, 2283, 1698, 2274, 2300, 1665, 1795, 1933, 2269, 1746, 1748, 1604, 2310, 2029, 1828, 2014, 2404, 2057, 2361, 1581, 2124, 2107, 2007, 1580, 1528, 1808, 1664, 2054, 2028, 2345, 1675, 1751, 2271, 1915, 1702, 1510, 1921, 2263, 2354, 2016, 1841, 1686, 2403, 1624, 1865, 1573, 1724, 1937, 2389, 1873, 1874, 2008, 2045, 2394, 1491, 1709, 1914, 1681, 2305, 2118, 1963, 2131, 2368, 2350, 2387, 2384, 2284, 1842, 2115, 1825, 1630, 1612, 1800, 1991, 2254, 1889, 1929, 1896, 1824, 2343, 1731, 2295, 2237, 1608, 1996, 2023, 1899, 1982, 1770, 2048, 1796, 2352, 1594, 1832, 1912, 1983, 1585, 1670, 2050, 1974, 1537, 1850, 2053, 2373, 1773, 2231, 1674, 2010, 2104, 1649, 2339, 2246, 2036, 1736, 1790, 1496, 1621, 2012, 1834, 1616, 1538, 1905, 1948, 1559, 2324, 2375, 2380, 1534, 1644, 1574, 2130, 2248, 2332, 1742, 1547, 2108, 1810, 2015, 1705, 2383, 1868, 1882, 1607, 1652, 1532, 1516, 2255, 2275, 1913, 1563, 2267, 2262, 1925, 2407, 1918, 2315, 2290, 1820, 2393, 2312, 1508, 1501, 2337, 1940, 1777, 2385, 1673, 1582, 2043, 1833, 1601, 1849, 1923, 1527, 1708, 2317, 1722, 2325, 1514, 1747, 1589, 2333, 1712, 1639, 1879, 1613, 1509, 1499, 1855, 2308, 2129, 1685, 1495, 2344, 1663, 1494, 2277, 2322, 1504, 1700, 1904, 1721, 1701, 1730, 2287, 2119, 1692, 1713, 1999, 1728, 2286, 2346, 2268, 2366, 1743, 1505, 1588, 1641, 2245, 1659, 1714, 2035, 2341, 1511, 1497, 1610, 2358, 2133, 2024, 2388, 1716, 1596, 1566, 1956, 1966, 1697, 1694, 1838, 1541, 1769, 1629, 1726, 1995, 1854, 1567, 1660, 1643, 2258, 1789, 1924, 2061, 2405, 2260, 2060, 1513, 1955, 1775, 1519, 2296, 1926, 1586, 1853, 1831, 2397, 1759, 1757, 1707, 1987, 1725, 1802, 1553, 1531, 1634, 1952, 2058, 1565, 1533, 1526, 2019, 2370, 2360, 2289, 2349, 1575, 1632, 1633, 2264, 1684, 2242, 1776, 1623, 2363, 1765, 2009, 1695, 1779, 1787, 1744, 1752, 1676, 1760, 1618, 2132, 1947, 2111, 2330, 2367, 1544, 2025, 2340, 2292, 1815, 1930, 1498, 1671, 1895, 2392, 2381, 2020, 1994, 1985, 2400, 1771, 1637, 2126, 1579, 1687, 2110, 2252, 1717, 1935, 1560, 1861, 2313, 2338, 1521, 1600, 1883, 1548, 1818, 1740, 2247, 2257, 2037, 2249, 2297, 1944, 2240, 1762, 1967, 1890, 1848, 2401, 1554, 1625, 2280, 2399, 2273, 2055, 1866, 2376, 1998, 1577, 2377, 2278, 1863, 2320, 2123, 2135, 2316, 1615, 1992, 2120, 2266, 1972, 1761, 2040, 1814, 1907, 1877, 1500, 1572, 1620, 1599, 1667, 1550, 1970, 1602, 1530              ,611, 1096, 522, 436, 412, 440, 439, 720, 430, 520, 435, 616, 463, 688, 1214, 415, 1078, 163, 421, 464, 461, 445, 607, 420, 442, 429, 584, 419, 610, 417, 714, 448, 581, 153, 1136, 1132, 444, 518, 422, 571, 619, 458, 718, 459, 1072, 574, 157, 721, 577, 713, 457, 583, 446, 1210, 460, 717, 432, 623, 519, 1082, 716, 411, 443, 580, 449, 418, 447, 462, 416, 160, 622, 465, 1100, 715, 521, 450, 414]
            drop_for_ps = split_df.index[split_df['UNIQUE_ID'].isin(l)].tolist()
            split_df = split_df.drop(drop_for_ps).reset_index(drop=True)


            split_df['id'] = split_df['UNIQUE_ID']
            split_df['partition'] = split_df['SPLIT'].apply(lambda split: partition_to_idx[split])

        self._split_array = split_df['partition'].values

        # y_array stores idx ids corresponding to location. Actual y labels are
        # tensors that are loaded separately.
        self._y_array = torch.from_numpy(split_df['id'].values)
        self._y_size = (self.img_dim, self.img_dim)

        self._metadata_fields = ['y']
        self._metadata_array = torch.from_numpy(split_df['id'].values)

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.get_label(idx)
        metadata = self.get_metadata(idx)
        return x, y, metadata

    def pad_m(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        pad_size = GRID_SIZE[self.country] - tensor.shape[-1]
        tensor = torch.nn.functional.pad(input=tensor, pad=(0, pad_size), value=0)
        return tensor

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def pad(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        # print("shape of tensor before padding",tensor.shape,tensor.shape[-1])
        num_samples = 40
        timestamps = tensor.shape[-1]
        if (timestamps < num_samples):
            return torch.nn.functional.pad(input=tensor, pad=(0, num_samples - timestamps), value=0)
        scores = np.ones((timestamps,))
        # print(scores)
        probabilities = self.softmax(scores)
        samples = np.random.choice(timestamps, size=num_samples, replace=False, p=probabilities)
        samples.sort()
        sampled_img_stack = tensor[:, :, :, samples]
        return sampled_img_stack

    def get_input(self, idx):
        """
        Returns X for a given idx.
        """
        loc_id = f'{self.y_array[idx]:06d}'
        if loc_id == "004660":  # problem in soutsudan one
            loc_id = "001603"  # random one
        images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))

        temp = images['s1'].astype(np.int64)
        s1 = []
        for t in range(temp.shape[-1]):  # (B x H x W x T) shape of the input image
            if np.any(temp[:, :, :, t]):
                s1.append(temp[:, :, :, t])
        temp = images['s2'].astype(np.int64)
        s2 = []
        for t in range(temp.shape[-1]):
            if np.any(temp[:, :, :, t]):
                s2.append(temp[:, :, :, t])
        temp = images['l8'].astype(np.int64)
        l8 = []
        for t in range(temp.shape[-1]):
            if np.any(temp[:, :, :, t]):
                l8.append(temp[:, :, :, t])
        #temp = images['planet'].astype(np.int64)
        #planet = []
        #for t in range(temp.shape[-1]):
        #    if np.any(temp[:, :, :, t]):
        #        planet.append(temp[:, :, :, t])

        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        l8 = np.asarray(l8)
        #planet = np.asarray(planet)

        s1 = torch.from_numpy(s1).permute(1, 0, 2, 3)
        s2 = torch.from_numpy(s2).permute(1, 0, 2, 3)
        l8 = torch.from_numpy(l8).permute(1, 0, 2, 3)
        try:
           planet = torch.from_numpy(planet).permute(1, 0, 2, 3)
        except:
           planet = torch.zeros((4, 184, 108, 108))

        if self.resize_planet:
            s1 = transforms.Resize(self.img_dim)(s1)
            s1 = s1.permute(0, 2, 3, 1)
            s2 = transforms.Resize(self.img_dim)(s2)
            s2 = s2.permute(0, 2, 3, 1)
            l8 = transforms.Resize(self.img_dim)(l8)
            l8 = l8.permute(0, 2, 3, 1)
            #planet = transforms.Resize(self.img_dim)(planet)
            #planet = planet.permute(0, 2, 3, 1)
        else:
            s1 = transforms.CenterCrop(PLANET_DIM)(s1)
            s1 = s1.permute(0, 2, 3, 1)
            s2 = transforms.CenterCrop(PLANET_DIM)(s2)
            s2 = s2.permute(0, 2, 3, 1)
            l8 = transforms.CenterCrop(PLANET_DIM)(l8)
            l8 = l8.permute(0, 2, 3, 1)
            #planet = transforms.CenterCrop(PLANET_DIM)(planet)
            #planet = planet.permute(0, 2, 3, 1)

        # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection
        if self.calculate_bands:
            ndvi_s2 = (s2[BANDS['s2']['10']['NIR']] - s2[BANDS['s2']['10']['RED']]) / (
                        s2[BANDS['s2']['10']['NIR']] + s2[BANDS['s2']['10']['RED']])
            ndvi_l8 = (l8[BANDS['l8']['8']['NIR']] - l8[BANDS['l8']['8']['RED']]) / (
                        l8[BANDS['l8']['8']['NIR']] + l8[BANDS['l8']['8']['RED']])
            #ndvi_planet = (planet[BANDS['planet']['4']['NIR']] - planet[BANDS['planet']['4']['RED']]) / (
            #            planet[BANDS['planet']['4']['NIR']] + planet[BANDS['planet']['4']['RED']])

            gcvi_s2 = (s2[BANDS['s2']['10']['NIR']] / s2[BANDS['s2']['10']['GREEN']]) - 1
            gcvi_l8 = (l8[BANDS['l8']['8']['NIR']] / l8[BANDS['l8']['8']['GREEN']]) - 1
            #gcvi_planet = (planet[BANDS['planet']['4']['NIR']]/planet[BANDS['planet']['4']['GREEN']]) - 1
            ## added this to remove nans
            ndvi_s2[(s2[BANDS['s2']['10']['NIR'], :, :, :] + s2[BANDS['s2']['10']['RED'], :, :, :]) == 0] = 0
            gcvi_s2[s2[BANDS['s2']['10']['GREEN'], :, :, :] == 0] = 0
            ndvi_l8[
                (l8[BANDS['l8']['8']['NIR'], :, :, :] + l8[BANDS['l8']['8']['RED'], :, :, :]) == 0] = 0
            gcvi_l8[l8[BANDS['l8']['8']['GREEN'], :, :, :] == 0] = 0
            #ndvi_planet[(planet[BANDS['planet']['4']['NIR'], :, :, :] + planet[BANDS['planet']['4']['RED'], :, :, :]) == 0] = 0
            #gcvi_planet[planet[BANDS['planet']['4']['GREEN'], :, :, :] == 0] = 0

        if self.normalize:
            s1 = self.normalization(s1, 's1')
            s2 = self.normalization(s2, 's2')
            l8 = self.normalization(l8, 'l8')
            #planet = self.normalization(planet, 'planet')

        # Concatenate calculated bands
        if self.calculate_bands:
            s2 = torch.cat((s2, torch.unsqueeze(ndvi_s2, 0), torch.unsqueeze(gcvi_s2, 0)), 0)
            #planet = torch.cat((planet, torch.unsqueeze(ndvi_planet, 0), torch.unsqueeze(gcvi_planet, 0)), 0)
            l8 = torch.cat((l8, torch.unsqueeze(ndvi_l8, 0), torch.unsqueeze(gcvi_l8, 0)), 0)

        s1 = self.pad(s1)
        s2 = self.pad(s2)
        l8 = self.pad(l8)
        #planet = self.pad(planet)

        if self._split_scheme in ['official', 'ghana', 'southsudan']:
            ## add code to get 32X32 from 64X64 using indexes
            grid_id = self.grid_id[idx].item()
            if (grid_id == 0):
                return {'s1': s1[:, 0:32, 0:32, :], 's2': s2[:, 0:32, 0:32, :],'l8': l8[:, 0:32, 0:32, :], 'planet': planet[:, 0:32, 0:32, :]}
            elif (grid_id == 1):
                return {'s1': s1[:, 32:64, 0:32, :], 's2': s2[:, 32:64, 0:32, :],'l8': l8[:, 32:64, 0:32, :], 'planet': planet[:, 32:64, 0:32, :]}
            elif (grid_id == 2):
                return {'s1': s1[:, 0:32, 32:64, :], 's2': s2[:, 0:32, 32:64, :],'l8': l8[:, 0:32, 32:64, :], 'planet': planet[:, 0:32, 32:64, :]}
            return {'s1': s1[:, 32:64, 32:64, :], 's2': s2[:, 32:64, 32:64, :],'l8': l8[:, 32:64, 32:64, :], 'planet': planet[:, 32:64, 32:64, :]}
        return {'s1': s1[self.s1_bands], 's2': s2[self.s2_bands], 'l8': l8[self.l8_bands], 'planet': l8[self.ps_bands]}

    def get_label(self, idx):
        """
        Returns y for a given idx.
        """
        loc_id = f'{self.y_array[idx]:06d}'
        if self._split_scheme in ['official', 'ghana', 'southsudan']:
            label = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))
            label = torch.from_numpy(label['truth'])
            ## added code
            grid_id = self.grid_id[idx].item()
            if (grid_id == 0):
                return label[0:32, 0:32]
            elif (grid_id == 1):
                return label[32:64, 0:32]
            elif (grid_id == 2):
                return label[0:32, 32:64]
            return label[32:64, 32:64]
        else:
            label = np.load(os.path.join(self.data_dir, self.country, f'truth@{self.truth_mask}m', f'{self.country}_{loc_id}.npz'))
            label = torch.from_numpy(np.expand_dims(label['crop_type'], 0))
            label = transforms.Resize(self.img_dim, interpolation=transforms.InterpolationMode.NEAREST)(label)[0]
        return label

    def get_dates(self, json_file):
        """
        Converts json dates into tensor containing dates
        """
        dates = np.array(json_file['dates'])
        dates = np.char.replace(dates, '-', '')
        dates = torch.from_numpy(dates.astype(np.int))
        return dates

    def get_metadata(self, idx):
        """
        Returns metadata for a given idx.
        Dates are returned as integers in format {Year}{Month}{Day}
        """
        if self._split_scheme == "cauvery":
            return None
        loc_id = f'{self.y_array[idx]:06d}'

        s1_json = json.loads(
            open(os.path.join(self.data_dir, self.country, 's1', f's1_{self.country}_{loc_id}.json'), 'r').read())
        s1 = self.get_dates(s1_json)

        s2_json = json.loads(
            open(os.path.join(self.data_dir, self.country, 's2', f's2_{self.country}_{loc_id}.json'), 'r').read())
        s2 = self.get_dates(s2_json)

        l8_json = json.loads(
            open(os.path.join(self.data_dir, self.country, 'l8', f'l8_{self.country}_{loc_id}.json'), 'r').read())
        l8 = self.get_dates(l8_json)

        planet_json = json.loads(
            open(os.path.join(self.data_dir, self.country, 'planet', f'planet_{self.country}_{loc_id}.json'),
                 'r').read())
        planet = self.get_dates(planet_json)

        # s1 = self.pad(s1)
        # s2 = self.pad(s2)
        # planet = self.pad(planet)
        s1 = self.pad_m(s1)
        s2 = self.pad_m(s2)
        planet = self.pad_m(planet)
        l8 = self.pad_m(l8)
        return {'s1': s1, 's2': s2, 'l8': l8, 'planet': planet}

    def normalization(self, grid, satellite):
        """ Normalization based on values defined in constants.py
        Args:
          grid - (tensor) grid to be normalized
          satellite - (str) describes source that grid is from ("s1" or "s2")
        Returns:
          grid - (tensor) a normalized version of the input grid
        """
        num_bands = grid.shape[0]
        means = MEANS[satellite][self.country]
        stds = STDS[satellite][self.country]
        grid = (grid - means[:num_bands].reshape(num_bands, 1, 1, 1)) / stds[:num_bands].reshape(num_bands, 1, 1, 1)

        if satellite not in ['s1', 's2', 'l8', 'planet']:
            raise ValueError("Incorrect normalization parameters")
        return grid

    def crop_segmentation_metrics(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert (y_true.shape == y_pred.shape)
        y_true = y_true.int()
        y_pred = y_pred.int()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        print('Macro Dice/ F1 score:', f1)
        print('Accuracy score:', acc)
        return f1, acc

    def eval(self, y_pred, y_true, metadata):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model
            - y_true (Tensor): Ground-truth values
            - metadata (Tensor): Metadata
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        f1, acc = self.crop_segmentation_metrics(y_true, y_pred)
        results = [f1, acc]
        results_str = f'Dice/ F1 score: {f1}, Accuracy score: {acc}'
        return results, results_str

    @property
    def normalize(self):
        """
        True if images will be normalized.
        """
        return self._normalize

    @property
    def country(self):
        """
        String containing the country pertaining to the dataset.
        """
        return self._country

    @property
    def resize_planet(self):
        """
        True if planet satellite imagery will be resized to other satellite sizes.
        """
        return self._resize_planet

    @property
    def calculate_bands(self):
        """
        True if aditional bands (NDVI and GCVI) will be calculated on the fly and appended
        """
        return self._calculate_bands
