""" PURPOSE: Init environment variables for the raytrace soft 
CATEGORY:
raytracing, data handling, os

INPUTS:
forceinit : force reinitialization of the variables
forcelibfile : force the path and filename of the shared object to
               test out. Useful when the user want to bypass the
               ssw default shared object. As an alternative, the
               user can also set the envvar RT_FORCELIBFILE.
forcelibthread : force the path and filename of the shared object 
                 using threadings. Note that the boost package 
                 abd boost_thread must be installed on your machine.
                 The libboost_thread.so should be present in the
                 shared library search path (LD_LIBRARY_PATH on unix and linux).
                 See the scraytrace user's guide for more information.

 DESCRIPTION:
  Defines different environment variables depending on the 
  operating system.
   RT_PATH : path where the IDL raytrace sources are
   RT_SOSUBPATH : subdirectory where the library are: this is
                  determined function of the OS and processor type
   RT_SOFILENAME : the filename of the library
   RT_RUNFROM : 'ssw' if run from ssw
                'local' if run from a local copy
   RT_FORCELIBFILE : Set to the path and filename of the shared
                     object in order to bypass the ssw precompiled one.
   RT_LIBFILE : contains the shared object path and filename.
   RT_LIBFILETHREAD : contains the path and filename of the shared object
                      used for multi-threading."""
def rtinitenv(forceinit=forceinit,forcelibfile=forcelibfile,forcelibthread=forcelibthread,help=help):
    