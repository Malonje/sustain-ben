import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# path = '/home/parichya/Documents/cauvery/'
path="/home/parichya/cauvery/cauvery/"
csv = pd.read_csv(os.path.join(path, 'cauvery_dataset.csv'))
# l=[1677, 1887, 1571, 1753, 2046, 1617, 1626, 1881, 1587, 1661, 1739, 2329, 2042, 1778, 1962, 2314, 1622, 2391, 2304, 1878, 2293, 1540, 2372, 1945, 2406, 2031, 1666, 1720, 2236, 2347, 1993, 1591, 1646, 1971, 1551, 2382, 1902, 2005, 1507, 2396, 1954, 1978, 2004, 1801, 1636, 2011, 2378, 1535, 1561, 2041, 2311, 1653, 2243, 2032, 2239, 1749, 1766, 1738, 1590, 1750, 2034, 1557, 1606, 1774, 1542, 1584, 1934, 1799, 1958, 1654, 1909, 1936, 1570, 1682, 1655, 1642, 1628, 1903, 2259, 1867, 1672, 1524, 1651, 2359, 2307, 2336, 1583, 2127, 1969, 2410, 1781, 2319, 1908, 2122, 1846, 1847, 2294, 2027, 1518, 1578, 1691, 1836, 1502, 1959, 1619, 1782, 2234, 1718, 1658, 2276, 1595, 1603, 1693, 1683, 2006, 2303, 1656, 1756, 1767, 2402, 2103, 1552, 1946, 1517, 1939, 2022, 2136, 1506, 1657, 1891, 1745, 1680, 1975, 1592, 1732, 2265, 1852, 1943, 1758, 1898, 2272, 2323, 1545, 1837, 2369, 1569, 2408, 1851, 2282, 1597, 2049, 2256, 1662, 2232, 1864, 1648, 1689, 1503, 1735, 1928, 2365, 1515, 1906, 1650, 1780, 1611, 2038, 1844, 2138, 2128, 1822, 1809, 1951, 1791, 1793, 2291, 1768, 1807, 2299, 1715, 1880, 1794, 1520, 2261, 2113, 1901, 1727, 2353, 2003, 1869, 1876, 2051, 1556, 2298, 1817, 2121, 2318, 1938, 1690, 1706, 2250, 1564, 2285, 2328, 1679, 1741, 2362, 1598, 1922, 1614, 2342, 1668, 1811, 1678, 1961, 1980, 2117, 2134, 2270, 1627, 1860, 2233, 1696, 1988, 1536, 1772, 2309, 1645, 1640, 1635, 2021, 2331, 1576, 2106, 2326, 2030, 1562, 2351, 1843, 1984, 1647, 1529, 1605, 2059, 1669, 1862, 2033, 1894, 2364, 1711, 2114, 1786, 1989, 2355, 1523, 1900, 1543, 1493, 1492, 2281, 2017, 2356, 1870, 1723, 1593, 1888, 2047, 2137, 1953, 1816, 2398, 1979, 2348, 1512, 2321, 2409, 2334, 2335, 1710, 2395, 1688, 2026, 1729, 2109, 2125, 1539, 1819, 2105, 2386, 2390, 2238, 1792, 2288, 1546, 1609, 1973, 1631, 1806, 2116, 2001, 1859, 1986, 1845, 1829, 2044, 2112, 1823, 1549, 1525, 2301, 2000, 1960, 1558, 1699, 1835, 2251, 1886, 1897, 2244, 2018, 1703, 1568, 1821, 2279, 2002, 1916, 2052, 1858, 2357, 2327, 2241, 1981, 2056, 1990, 1719, 2374, 2235, 2379, 2253, 1555, 2039, 2013, 1927, 1522, 1805, 1638, 2306, 2302, 1917, 1704, 1997, 1957, 2371, 2283, 1698, 2274, 2300, 1665, 1795, 1933, 2269, 1746, 1748, 1604, 2310, 2029, 1828, 2014, 2404, 2057, 2361, 1581, 2124, 2107, 2007, 1580, 1528, 1808, 1664, 2054, 2028, 2345, 1675, 1751, 2271, 1915, 1702, 1510, 1921, 2263, 2354, 2016, 1841, 1686, 2403, 1624, 1865, 1573, 1724, 1937, 2389, 1873, 1874, 2008, 2045, 2394, 1491, 1709, 1914, 1681, 2305, 2118, 1963, 2131, 2368, 2350, 2387, 2384, 2284, 1842, 2115, 1825, 1630, 1612, 1800, 1991, 2254, 1889, 1929, 1896, 1824, 2343, 1731, 2295, 2237, 1608, 1996, 2023, 1899, 1982, 1770, 2048, 1796, 2352, 1594, 1832, 1912, 1983, 1585, 1670, 2050, 1974, 1537, 1850, 2053, 2373, 1773, 2231, 1674, 2010, 2104, 1649, 2339, 2246, 2036, 1736, 1790, 1496, 1621, 2012, 1834, 1616, 1538, 1905, 1948, 1559, 2324, 2375, 2380, 1534, 1644, 1574, 2130, 2248, 2332, 1742, 1547, 2108, 1810, 2015, 1705, 2383, 1868, 1882, 1607, 1652, 1532, 1516, 2255, 2275, 1913, 1563, 2267, 2262, 1925, 2407, 1918, 2315, 2290, 1820, 2393, 2312, 1508, 1501, 2337, 1940, 1777, 2385, 1673, 1582, 2043, 1833, 1601, 1849, 1923, 1527, 1708, 2317, 1722, 2325, 1514, 1747, 1589, 2333, 1712, 1639, 1879, 1613, 1509, 1499, 1855, 2308, 2129, 1685, 1495, 2344, 1663, 1494, 2277, 2322, 1504, 1700, 1904, 1721, 1701, 1730, 2287, 2119, 1692, 1713, 1999, 1728, 2286, 2346, 2268, 2366, 1743, 1505, 1588, 1641, 2245, 1659, 1714, 2035, 2341, 1511, 1497, 1610, 2358, 2133, 2024, 2388, 1716, 1596, 1566, 1956, 1966, 1697, 1694, 1838, 1541, 1769, 1629, 1726, 1995, 1854, 1567, 1660, 1643, 2258, 1789, 1924, 2061, 2405, 2260, 2060, 1513, 1955, 1775, 1519, 2296, 1926, 1586, 1853, 1831, 2397, 1759, 1757, 1707, 1987, 1725, 1802, 1553, 1531, 1634, 1952, 2058, 1565, 1533, 1526, 2019, 2370, 2360, 2289, 2349, 1575, 1632, 1633, 2264, 1684, 2242, 1776, 1623, 2363, 1765, 2009, 1695, 1779, 1787, 1744, 1752, 1676, 1760, 1618, 2132, 1947, 2111, 2330, 2367, 1544, 2025, 2340, 2292, 1815, 1930, 1498, 1671, 1895, 2392, 2381, 2020, 1994, 1985, 2400, 1771, 1637, 2126, 1579, 1687, 2110, 2252, 1717, 1935, 1560, 1861, 2313, 2338, 1521, 1600, 1883, 1548, 1818, 1740, 2247, 2257, 2037, 2249, 2297, 1944, 2240, 1762, 1967, 1890, 1848, 2401, 1554, 1625, 2280, 2399, 2273, 2055, 1866, 2376, 1998, 1577, 2377, 2278, 1863, 2320, 2123, 2135, 2316, 1615, 1992, 2120, 2266, 1972, 1761, 2040, 1814, 1907, 1877, 1500, 1572, 1620, 1599, 1667, 1550, 1970, 1602, 1530              ,611, 1096, 522, 436, 412, 440, 439, 720, 430, 520, 435, 616, 463, 688, 1214, 415, 1078, 163, 421, 464, 461, 445, 607, 420, 442, 429, 584, 419, 610, 417, 714, 448, 581, 153, 1136, 1132, 444, 518, 422, 571, 619, 458, 718, 459, 1072, 574, 157, 721, 577, 713, 457, 583, 446, 1210, 460, 717, 432, 623, 519, 1082, 716, 411, 443, 580, 449, 418, 447, 462, 416, 160, 622, 465, 1100, 715, 521, 450, 414]
# drop_for_ps = csv.index[csv['UNIQUE_ID'].isin(l)].tolist()
# csv = csv.drop(drop_for_ps).reset_index(drop=True)
# filenames = csv[csv['SPLIT'] == 'train']['UNIQUE_ID'].values
filenames = csv[csv['SPLIT'] == 'val']['UNIQUE_ID'].values

# dataset = [np.load(os.path.join(path, 'npy', f'cauvery_{fn:06d}.npz')) for fn in filenames]
# print("len(dataset)", len(dataset))
# l8 = []
# s1 = []
l8 = []
files=os.listdir(path+'npy/')
# ps = []
for i in tqdm(range(0,len(filenames))):
    fn=filenames[i]
    im_l8 = np.load(path+'npy/'+f'cauvery_{fn:06d}.npz')['l8']
    # print(im['l8'].shape, im['s1'].shape, im['s2'].shape)
    im_l8 = im_l8.transpose((3, 0, 1, 2))
    for t in range(im_l8.shape[0]):

        # im_l8 = im['l8'].transpose((3, 0, 1, 2))
        # im_s1 = im['s1'].transpose((3, 0, 1, 2))
        # im_s2 = im['s2'].transpose((3, 0, 1, 2))
        # print(im_p.shape)
        if np.any(im_l8[t]):
            l8.append(im_l8[t])
        # if np.any(im_s1[t]):
        #     s1.append(im_s1[t])
        # if np.any(im_s2[t]):
        #     s2.append(im_s2[t])
        # if np.any(im_p[t]):
        #     ps.append(im_p[t])
l8 = np.array(l8)
# s1 = np.array(s1)
# s1 = np.array(s1)
# ps = np.array(ps)
print("L8", l8.shape)
print(np.mean(l8, (0, 2, 3)))
print(np.std(l8, (0, 2, 3)))
print()
# print("S1", s1.shape)
# print(np.mean(s1, (0, 2, 3)))
# print(np.std(s1, (0, 2, 3)))
# print()
# print("S1", s1.shape)
# print(np.mean(s1, (0, 2, 3)))
# print(np.std(s1, (0, 2, 3)))
# print()
# print("PS", ps.shape)
# print(np.mean(ps, (0, 2, 3)))
# print(np.std(ps, (0, 2, 3)))
# print()


# L8 (16002, 8, 11, 11)
# on train:
# [12323.99321417 12618.39105649 13599.00590732 13469.41012126
#  19246.94627376 14150.09082232 11900.22527039 30151.18871195]
# [ 9900.16857204  9776.42869284  9109.375738    9120.84374404
#   8959.01997492  6547.22589968  5485.36549631 19514.8944211 ]
### mean around .2 and val: 0.08-0.15
 

# on val:
# L8 (4052, 8, 11, 11)
# [12221.69553858 12516.27836269 13501.42842225 13375.85782146
#  19117.57466979 14134.92964397 11893.74423609 29961.74466644]
# [ 9795.26642324  9675.19681508  9031.23604828  9044.8099439
#   8965.87766943  6652.39441641  5578.73634104 19770.20918013]
