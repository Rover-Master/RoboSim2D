# 2D Robotic Simulation Framework

Authors: [Yuxuan Zhang](mailto:robotics@z-yx.cc), Adnan Abdullah

### Basic Examples

#### Random Walk

<table style="width: 720px">
  <tr>
    <th style="text-align: center">Attempt 01</th>
    <th style="text-align: center">Attempt 02</th>
    <th style="text-align: center">Attempt 03</th>
    <th style="text-align: center">Attempt 04</th>
  </tr>
  <tr>
    <td><image src="doc/RandomWalk-01.png"></td>
    <td><image src="doc/RandomWalk-02.png"></td>
    <td><image src="doc/RandomWalk-03.png"></td>
    <td><image src="doc/RandomWalk-04.png"></td>
  </tr>
  <tr>
    <td style="text-align: center" colspan=4>
        <code>python3 -m simulation.RandomWalk data/world -v</code>
    </td>
  </tr>
</table>

#### Bug0

<table style="width: 720px">
  <tr>
    <th style="text-align: center">Left Turn Variant</th>
    <th style="text-align: center">Right Turn Variant</th>
  </tr>
  <tr>
    <td><image src="doc/Bug0L.png"></td>
    <td><image src="doc/Bug0R.png"></td>
  </tr>
  <tr>
    <td style="text-align: center"><code>simulation.Bug0L</code></td>
    <td style="text-align: center"><code>simulation.Bug0R</code></td>
  </tr>
</table>

#### Bug1

<table style="width: 720px">
  <tr>
    <th style="text-align: center">Left Turn Variant</th>
    <th style="text-align: center">Right Turn Variant</th>
  </tr>
  <tr>
    <td><image src="doc/Bug1L.png"></td>
    <td><image src="doc/Bug1R.png"></td>
  </tr>
  <tr>
    <td style="text-align: center"><code>simulation.Bug1L</code></td>
    <td style="text-align: center"><code>simulation.Bug1R</code></td>
  </tr>
</table>

#### Bug2

<table style="width: 720px">
  <tr>
    <th style="text-align: center">Left Turn Variant</th>
    <th style="text-align: center">Right Turn Variant</th>
  </tr>
  <tr>
    <td><image src="doc/Bug2L.png"></td>
    <td><image src="doc/Bug2R.png"></td>
  </tr>
  <tr>
    <td style="text-align: center"><code>simulation.Bug2L</code></td>
    <td style="text-align: center"><code>simulation.Bug2R</code></td>
  </tr>
</table>

#### WaveFront

```sh
python3 -m wavefront.WaveFront data/world -v
```

### Detailed Usage

```sh
python3 -m <module> <world>
    [-h] [--prefix PREFIX] [--src SRC] [--dst DST]
    [-t THRESHOLD] [-r RADIUS] [-M MAX_TRAVEL]
    [-s SCALE] [--dpi-scale DPI_SCALE] [-v] [--no-wait]
    [--line-width LINE_WIDTH] [--line-color LINE_COLOR]
    [-R RESOLUTION] [-S SLICE] [--debug]

modules:
  simulation.RandomWalk
  simulation.Bug0L
  simulation.Bug0R
  simulation.Bug1L
  simulation.Bug1R
  simulation.Bug2L
  simulation.Bug2R
  wavefront.WaveFront

positional arguments:
  world                 Path to the map file

options:
  -h, --help            show this help message and exit
  --prefix PREFIX       Destination file or folder
  --src SRC             Simulation Starting Location
  --dst DST             Simulation Destination
  -t, --threshold THRESHOLD
                        Threshold distance near the target
  -r, --radius RADIUS   Collision radius for the robot, in meters
  -M, --max-travel MAX_TRAVEL
                        Max travel distance before aborting the simulation
  -s, --scale SCALE     Scale of viewport, relative to internal occupancy grid
  --dpi-scale DPI_SCALE
                        DPI Scale for UI elements (lines, markers, text)
  -v, --visualize       Enable visualization
  --no-wait             Do not wait for key stroke after simulation is complete, effective only with the -v flag
  --line-width LINE_WIDTH
                        Line Width for Visualization, in Meters
  --line-color LINE_COLOR
                        Color of the trajectory line, in "b,g,r" format
  -R, --resolution RESOLUTION
                        World pixel resolution in meters (m/px)
  -S, --slice SLICE     Output Image Slice in "x,y,w,h" format
  --debug               Toggle debug tools
```
