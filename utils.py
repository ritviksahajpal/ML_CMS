import os, calendar, pdb, numpy, errno, logging, pandas, csv, sys, netCDF4

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    """
    Fast version of numpy genfromtxt
    code from here: http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy/8964779#8964779
    """
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = numpy.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def open_or_die(path_file, perm='r', csv_header=0, skiprows=0, delimiter=' ', mask_val=-9999.0, fl_format=''):
    """
    Open file or quit gracefully
    :param path_file: Path of file to open
    :param perm: Permissions with which to open file. Default is read-only
    :param csv_header:
    :param skiprows:
    :param delimiter:
    :param mask_val:
    :param fl_format: Special code for some file openings
    :return: Handle to file (netCDF), or dataframe (csv) or numpy array
    """

    try:
        if os.path.splitext(path_file)[1] == '.nc':
            hndl = netCDF4.Dataset(path_file, perm, format='NETCDF4')
            return hndl
        elif os.path.splitext(path_file)[1] == '.csv':
            df = pandas.read_csv(path_file, header=csv_header)
            return df
        elif os.path.splitext(path_file)[1] == '.xlsx' or os.path.splitext(path_file)[1] == '.xls':
            df = pandas.ExcelFile(path_file, header=csv_header)
            return df
        elif os.path.splitext(path_file)[1] == '.asc':
            data = iter_loadtxt(path_file, delimiter=delimiter, skiprows=skiprows)
            data = numpy.ma.masked_values(data, mask_val)
            return data
        elif os.path.splitext(path_file)[1] == '.txt':
            data = iter_loadtxt(path_file, delimiter=delimiter, skiprows=skiprows)
            data = numpy.ma.masked_values(data, mask_val)
            return data
        else:
            logging.error('Invalid file type ' + os.path.splitext(path_file)[1])
            sys.exit(0)
    except:
        logging.error('Error opening file ' + path_file)
        sys.exit(0)


def make_dir_if_missing(d):
    """
    Create directory if not present, else do nothing
    :param d: Path of directory to create
    :return: Nothing, side-effect: create directory
    """
    try:
        os.makedirs(d)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def go_higher_dir_levels(path_to_dir, level=0):
    """
    Gien directory path, go up number of levels defined by level
    :param path_to_dir:
    :param level:
    :return:
    """
    up_dir = path_to_dir

    for lev in range(level):
        up_dir = os.path.dirname(path_to_dir)

    return up_dir
