fail = False
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print('Voltools viewer requires PyQt5')
    fail = True

try:
    import pycuda
    from pycuda.compiler import DynamicSourceModule
    import pycuda.driver as cuda
    import pycuda.gpuarray as ga
    import pycuda.gl
except ImportError:
    print('Voltools viewer requires Pycuda with GL enabled.')
    fail = True

try:
    import os

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib.pyplot as plt

    import numpy as np
    from contextlib import contextmanager

    from OpenGL.GL import *
    from OpenGL.GL import shaders

    import warnings
    import sys
    import timeit
    from pyrr import Matrix44
    import mrcfile as mrc
except ImportError:
    print()
    fail = True

if fail:
    raise SystemExit('Imports failed, viewer will not be imported.')

PANEL_WIDTH = 300


class VolumeViewerWindow(QMainWindow):

    def __init__(self, data):
        super(VolumeViewerWindow, self).__init__(flags=Qt.Window)

        # app icon
        self.appIcon = QIcon()
        self.setWindowIcon(self.appIcon)

        # set the widget
        self.app = VolumeViewer(self, data)
        self.setCentralWidget(self.app)

        # set size
        if hasattr(data, 'shape'):
            self.resize(data.shape[1] + PANEL_WIDTH, data.shape[2])
            self.center()

    def center(self):
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def closeEvent(self, event):
        # close
        event.accept()

class VolumeViewer(QWidget):

    def __init__(self, parent, data):
        super(VolumeViewer, self).__init__(flags=Qt.Widget, parent=parent)

        self.data = data
        self.histogram = np.histogram(self.data[0], bins=256)[0]

        self.mainLayout = QHBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        # Operator panel
        self.operator = OperatorPanel(parent=self)
        self.mainLayout.addWidget(self.operator, stretch=0)

        # GL Viewport
        self.canvas = Renderer(data, parent=self)
        self.mainLayout.addWidget(self.canvas)

        self.setLayout(self.mainLayout)

class TooltipSlider(QSlider):

    def __init__(self, maximum, minimum=0, start=0):
        super(TooltipSlider, self).__init__()

        # Slider settings
        self.setRange(minimum, maximum - 1)
        self.setSingleStep(1)
        self.setPageStep(0) # disable PageUp/PageDown
        self.setTracking(True) # emits valueChanged() while dragging
        self.setOrientation(Qt.Horizontal)

        self.setValue(start)

class OperatorPanel(QWidget):

    def __init__(self, parent):
        super(OperatorPanel, self).__init__(flags=Qt.Widget, parent=parent)

        # pointer to data for easier access
        self.data = parent.data
        self.histogram = parent.histogram
        self.current_slice = 0

        self.mainLayout = QVBoxLayout()

        # Volume information
        self.infoBox = QGroupBox('Information')
        self.infoBox.mainLayout = QFormLayout(self.infoBox)
        self.infoBox.mainLayout.addRow(QLabel('Dimensions:'), QLabel(str(self.data.shape)))
        self.infoBox.mainLayout.addRow(QLabel('Precision:'), QLabel('{}, {} bytes per voxel'
                                                                    .format(self.data.dtype.name,
                                                                            self.data.dtype.itemsize)))
        self.infoBox.mainLayout.addRow(QLabel('Size:'), QLabel('{:.3f} MB'.format(self.data.nbytes / 1024 ** 2)))
        self.infoBox.volumeStats = QLabel('{:.3f} / {:.3f} / {:.3f}'.format(self.data.min(), self.data.max(),
                                                                            self.data.mean()))
        self.infoBox.mainLayout.addRow(QLabel('Volume Min/Max/Mean:'), self.infoBox.volumeStats)
        self.infoBox.sliceStats = QLabel('{:.3f} / {:.3f} / {:.3f}'.format(self.data[self.current_slice].min(),
                                                                           self.data[self.current_slice].max(),
                                                                           self.data[self.current_slice].mean()))
        self.infoBox.mainLayout.addRow(QLabel('Slice Min/Max/Mean:'), self.infoBox.sliceStats)
        self.infoBox.saveButton = QPushButton('Save current volume')
        self.infoBox.mainLayout.addRow(self.infoBox.saveButton)

        ## Histogram
        self.hist_fig = plt.figure(figsize=(2, 1))
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.infoBox.mainLayout.addRow(self.hist_canvas)

        self.mainLayout.addWidget(self.infoBox)

        # CUDA Panel
        self.cudaBox = QGroupBox('CUDA')
        self.cudaBox.mainLayout = QFormLayout(self.cudaBox)

        self.cudaBox.slider_rot_x = TooltipSlider(91, -91)
        self.cudaBox.label_rot_x = QLabel('0')
        self.cudaBox.mainLayout.addRow('Rotation around X:', self.cudaBox.label_rot_x)
        self.cudaBox.mainLayout.addRow(self.cudaBox.slider_rot_x)

        self.cudaBox.slider_rot_y = TooltipSlider(91, -91)
        self.cudaBox.label_rot_y = QLabel('0')
        self.cudaBox.mainLayout.addRow('Rotation around Y:', self.cudaBox.label_rot_y)
        self.cudaBox.mainLayout.addRow(self.cudaBox.slider_rot_y)

        self.cudaBox.slider_rot_z = TooltipSlider(91, -91)
        self.cudaBox.label_rot_z = QLabel('0')
        self.cudaBox.mainLayout.addRow('Rotation around Z:', self.cudaBox.label_rot_z)
        self.cudaBox.mainLayout.addRow(self.cudaBox.slider_rot_z)

        self.cudaBox.edit_trans_x = QLineEdit('0')
        self.cudaBox.mainLayout.addRow(QLabel('Translation X:'))
        self.cudaBox.mainLayout.addRow(self.cudaBox.edit_trans_x)

        self.cudaBox.edit_trans_y = QLineEdit('0')
        self.cudaBox.mainLayout.addRow(QLabel('Translation Y:'))
        self.cudaBox.mainLayout.addRow(self.cudaBox.edit_trans_y)

        self.cudaBox.edit_trans_z = QLineEdit('0')
        self.cudaBox.mainLayout.addRow(QLabel('Translation Z:'))
        self.cudaBox.mainLayout.addRow(self.cudaBox.edit_trans_z)

        self.cudaBox.slider_scale_x = TooltipSlider(5, -5, start=1)
        self.cudaBox.label_scale_x = QLabel('1')
        self.cudaBox.mainLayout.addRow('Scale X', self.cudaBox.label_scale_x)
        self.cudaBox.mainLayout.addRow(self.cudaBox.slider_scale_x)

        self.cudaBox.slider_scale_y = TooltipSlider(5, -5, start=1)
        self.cudaBox.label_scale_y = QLabel('1')
        self.cudaBox.mainLayout.addRow('Scale Y', self.cudaBox.label_scale_y)
        self.cudaBox.mainLayout.addRow(self.cudaBox.slider_scale_y)

        self.cudaBox.slider_scale_z = TooltipSlider(5, -5, start=1)
        self.cudaBox.label_scale_z = QLabel('1')
        self.cudaBox.mainLayout.addRow('Scale Z', self.cudaBox.label_scale_z)
        self.cudaBox.mainLayout.addRow(self.cudaBox.slider_scale_z)

        self.cudaBox.centeredCheckbox = QCheckBox('Transform around center of tomogram')
        self.cudaBox.centeredCheckbox.setChecked(True)
        self.cudaBox.mainLayout.addRow(self.cudaBox.centeredCheckbox)

        self.cudaBox.rotateButton = QPushButton('Transform')
        self.cudaBox.mainLayout.addRow(self.cudaBox.rotateButton)
        self.mainLayout.addWidget(self.cudaBox)

        # GL Panel
        self.glBox = QGroupBox('GL')
        self.glBox.mainLayout = QFormLayout(self.glBox)
        self.glBox.sliceLabel = QLabel('0')
        self.glBox.mainLayout.addRow(QLabel('Slice:'), self.glBox.sliceLabel)
        self.glBox.sliceSlider = TooltipSlider(self.data.shape[0])
        self.glBox.mainLayout.addRow(self.glBox.sliceSlider)
        self.mainLayout.addWidget(self.glBox)

        # Stretch space
        self.mainLayout.addStretch()

        # signal connections for transform sliders
        self.cudaBox.slider_rot_x.valueChanged.connect(lambda: self.cudaBox.label_rot_x.
                                                       setText('{}'.format(self.cudaBox.slider_rot_x.value())))
        self.cudaBox.slider_rot_y.valueChanged.connect(lambda: self.cudaBox.label_rot_y.
                                                       setText('{}'.format(self.cudaBox.slider_rot_y.value())))
        self.cudaBox.slider_rot_z.valueChanged.connect(lambda: self.cudaBox.label_rot_z.
                                                       setText('{}'.format(self.cudaBox.slider_rot_z.value())))

        self.cudaBox.slider_scale_x.valueChanged.connect(lambda: self.cudaBox.label_scale_x.
                                                         setText('{}'.format(self.cudaBox.slider_scale_x.value())))
        self.cudaBox.slider_scale_y.valueChanged.connect(lambda: self.cudaBox.label_scale_y.
                                                         setText('{}'.format(self.cudaBox.slider_scale_y.value())))
        self.cudaBox.slider_scale_z.valueChanged.connect(lambda: self.cudaBox.label_scale_z.
                                                         setText('{}'.format(self.cudaBox.slider_scale_z.value())))

        # rest of connections
        self.glBox.sliceSlider.valueChanged.connect(self.on_slice_slider)
        self.cudaBox.rotateButton.released.connect(self.on_cuda_rotate)
        self.infoBox.saveButton.pressed.connect(self.on_save)

        # final set
        self.setLayout(self.mainLayout)
        self.setFixedWidth(PANEL_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # plot histogram
        self.plot_histogram()

    def on_save(self):
        name = QFileDialog.getSaveFileName(self, 'Save File')
        if not name[0]:
            return

        self.parent().canvas.save_volume(name[0])

    def on_cuda_rotate(self):

        around_center = bool(self.cudaBox.centeredCheckbox.isChecked())

        rotations = [float(self.cudaBox.slider_rot_x.value()),
                     float(self.cudaBox.slider_rot_y.value()),
                     float(self.cudaBox.slider_rot_z.value())]

        scales = [float(self.cudaBox.slider_scale_x.value()),
                  float(self.cudaBox.slider_scale_y.value()),
                  float(self.cudaBox.slider_scale_z.value())]

        translations = [float(self.cudaBox.edit_trans_x.text()),
                        float(self.cudaBox.edit_trans_y.text()),
                        float(self.cudaBox.edit_trans_z.text())]

        m_translate = Matrix44.from_translation(translations, dtype=np.float32)
        m_scale = Matrix44.from_scale(scales, dtype=np.float32)

        m_rotation_x = Matrix44.from_x_rotation(np.deg2rad(rotations[0]), dtype=np.float32)
        m_rotation_y = Matrix44.from_y_rotation(np.deg2rad(rotations[1]), dtype=np.float32)
        m_rotation_z = Matrix44.from_z_rotation(np.deg2rad(rotations[2]), dtype=np.float32)

        m_rotation = m_rotation_x * m_rotation_y * m_rotation_z

        center_point = np.divide(self.data.shape[::-1], 2)
        m_pretranslation = Matrix44.from_translation(center_point, dtype=np.float32)
        m_posttranslation = Matrix44.from_translation(-1 * center_point, dtype=np.float32)

        # pyrr library's Matrix44 is transposed inside, so the order of dot products is left to right
        # also *-operator is overloaded to dotproduct instead of elementwise mult
        if around_center:
            m_transform = m_pretranslation * m_scale * m_rotation * m_posttranslation * m_translate
        else:
            m_transform = m_scale * m_rotation * m_translate

        # debug printout should be transposed
        # np.set_printoptions(precision=3)
        # print(m_transform.T)

        self.parent().canvas.cuda_rotate(m_transform)

    def on_slice_slider(self, value):
        self.current_slice = value
        self.glBox.sliceLabel.setText(str(value))
        self.infoBox.sliceStats.setText('{:.3f} / {:.3f} / {:.3f}'.format(self.data[self.current_slice].min(),
                                                                          self.data[self.current_slice].max(),
                                                                          self.data[self.current_slice].mean()))
        self.plot_histogram()
        self.parent().canvas.change_slice(self.current_slice)

    def plot_histogram(self):
        # clear previous
        self.hist_fig.clear()

        # compute histogram
        data = self.data[self.current_slice]
        # hist = histogram1d(data, bins=60, range=(data.min(), data.max()+0.001))
        self.histogram = np.histogram(data, bins=256)[0]

        # draw
        plt.tight_layout()
        plt.axis('off')

        # if not hasattr(self, 'ax'):
        self.ax = self.hist_fig.add_subplot(111)
        self.ax.plot(self.histogram)
        self.hist_canvas.draw()

class Renderer(QOpenGLWidget):

    def __init__(self, data, **kwargs):
        super(Renderer, self).__init__(**kwargs)

        fmt = QSurfaceFormat()
        fmt.setVersion(4, 2)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(fmt)

        self.data = data

    def initializeGL(self):

        # PyCUDA kernels and functions
        import pycuda.gl.autoinit
        os.environ['PATH'] += ';' + r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + r'/kernels.cu', 'r') as f:
            cudaCode = f.read()

        self.kernels = DynamicSourceModule(cudaCode, no_extern_c=True,
                                           options=['-O3', '--compiler-options', '-Wall', '-rdc=true', '-lcudadevrt'],
                                           cuda_libdir='C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64',
                                           include_dirs=['C:\\Projects\\ml\\experiments\\volume-viewer'])

        # Init shaders
        VERT_SHADER_SRC = '''
        #version 420 core
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;

        layout (location=0) in vec2 a_position;
        layout (location=1) in vec2 a_texcoord;

        out vec2 v_texcoord;

        void main() {
            v_texcoord = a_texcoord;
            gl_Position = u_projection * u_view * u_model * vec4(a_position, 0.0, 1.0);
        }
        '''
        FRAG_SHADER_SRC = '''
        #version 420 core

        uniform sampler2D u_texture;
        in vec2 v_texcoord;

        uniform sampler1D u_colormap;
        uniform float u_min_i;
        uniform float u_max_i;

        out vec4 FragColor;

        void main() {
            float val = texture(u_texture, v_texcoord).x;
            float norm_val = (val - u_min_i) / (u_max_i - u_min_i);

            if (val == 0.0f) {
                FragColor = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
                FragColor = texture(u_colormap, norm_val); 
            }
        }
        '''

        v_shader = shaders.compileShader(VERT_SHADER_SRC, GL_VERTEX_SHADER)
        f_shader = shaders.compileShader(FRAG_SHADER_SRC, GL_FRAGMENT_SHADER)

        self.prog = shaders.compileProgram(v_shader, f_shader)
        glUseProgram(self.prog)

        # Store uniform locations
        self.prog.uniforms = {'u_model': glGetUniformLocation(self.prog, 'u_model'),
                              'u_view': glGetUniformLocation(self.prog, 'u_view'),
                              'u_projection': glGetUniformLocation(self.prog, 'u_projection'),
                              'u_texture': glGetUniformLocation(self.prog, 'u_texture'),
                              'u_colormap': glGetUniformLocation(self.prog, 'u_colormap'),
                              'u_min_i': glGetUniformLocation(self.prog, 'u_min_i'),
                              'u_max_i': glGetUniformLocation(self.prog, 'u_max_i'),
                              }

        # Render quad
        self.quad = np.array([  # x, y, tex_x, tex_y
            1, 1, 1.0, 1.0,
            1, -1, 1.0, 0.0,
            -1, -1, 0.0, 0.0,
            -1, 1, 0.0, 1.0
        ], np.float32)
        self.indices = np.array([0, 1, 3, 1, 2, 3], np.uint32)

        # VBO, VAO, EBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.quad.nbytes, self.quad, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # location = 0, a_position
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * self.quad.strides[0], None)
        glEnableVertexAttribArray(0)

        # location = 1, a_texcoord
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * self.quad.strides[0],
                              ctypes.c_void_p(2 * self.quad.strides[0]))
        glEnableVertexAttribArray(1)

        # Alpha blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        # Depth test disabled, we don't need
        glDisable(GL_DEPTH_TEST)

        # MVP uniforms
        self.model = np.eye(4, dtype=np.float32)
        self.view = np.eye(4, dtype=np.float32)
        self.projection = Matrix44.orthogonal_projection(-1, 1, -1, 1, -1000, 1000, dtype=np.float32)

        glUniformMatrix4fv(self.prog.uniforms['u_model'], 1, GL_FALSE, self.model)
        glUniformMatrix4fv(self.prog.uniforms['u_view'], 1, GL_FALSE, self.view)
        glUniformMatrix4fv(self.prog.uniforms['u_projection'], 1, GL_FALSE, self.projection)

        # Slice params
        self.norm_slice = 0
        self.slice = 0
        slice_low, slice_high = np.percentile(self.data[self.slice], (0.5, 99.5))
        glUniform1f(self.prog.uniforms['u_min_i'], slice_low)
        glUniform1f(self.prog.uniforms['u_max_i'], slice_high)

        # Colormap
        self.color_ramp = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        self.colormap = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.colormap)
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, len(self.color_ramp), 0, GL_RGBA, GL_FLOAT, self.color_ramp)
        glUniform1i(self.prog.uniforms['u_colormap'], 0)
        glBindTexture(GL_TEXTURE_1D, 0)

        # Init (empty) coeffs texture
        self.coeff_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.coeff_tex)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        t_d, t_h, t_w = self.data.shape
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, t_w, t_h, t_d, 0, GL_RED, GL_FLOAT, None)
        self.coeff_tex_cuda = pycuda.gl.RegisteredImage(int(self.coeff_tex), GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, 0)

        # Run cuda init routine (prefilter, texture binding and populating the coeffs texture)
        self.cuda_prefilter()

        # Working buffer - this is where rotated volume will be stored
        # PBO: Pixel Buffer Object
        self.volume_buf = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.volume_buf)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.data.nbytes, None, GL_DYNAMIC_COPY)
        self.volume_buf_cuda = pycuda.gl.RegisteredBuffer(int(self.volume_buf))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # 2D texture which the render quad will use
        self.render_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.render_tex)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, t_w, t_h, 0, GL_RED, GL_FLOAT, None)
        glUniform1i(self.prog.uniforms['u_texture'], 1)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Populate PBO and 2D texture by running the rotation routine with identity matrix
        # (should also be faster than copying original volume from CPU)
        self.cuda_rotate(np.eye(4, dtype=np.float32))

    def paintGL(self):
        # print('paintGL')

        # clear
        glClearColor(0.3, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # textures, unit numbers should be consistent with set uniforms for u_texture and u_colormap
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, self.colormap)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.render_tex)

        # render
        glUseProgram(self.prog)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # ### DEMO
        # # Random Z  rotation
        # self.parent().operator.cudaBox.slider_rot_z.setValue(np.random.randint(-90, 90))
        # # Random slice
        # self.parent().operator.glBox.sliceSlider.setValue(np.random.randint(0, self.data.shape[0]))
        # self.parent().operator.on_cuda_rotate()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        # preserve aspect ratio
        ar = width / height

        if ar >= 1:
            self.projection = Matrix44.orthogonal_projection(-1 * ar, 1 * ar, -1, 1, -1000, 1000, dtype=np.float32)
        else:
            self.projection = Matrix44.orthogonal_projection(-1, 1, -1 / ar, 1 / ar, -1000, 1000, dtype=np.float32)

        glUniformMatrix4fv(self.prog.uniforms['u_projection'], 1, GL_FALSE, self.projection)

    def cuda_prefilter(self):

        def PowTwoDivider(n: int):
            if n == 0:
                return 0

            divider = 1
            while (n & divider) == 0:
                divider <<= 1

            return divider

        # Prefilter done only once, so no need do prepared calls, get_function and use it
        SamplesToCoefficients3DX = self.kernels.get_function('SamplesToCoefficients3DX')
        SamplesToCoefficients3DY = self.kernels.get_function('SamplesToCoefficients3DY')
        SamplesToCoefficients3DZ = self.kernels.get_function('SamplesToCoefficients3DZ')

        # https://lists.tiker.net/pipermail/pycuda/2009-November/001913.html
        gca = ga.to_gpu(self.data)
        depth, height, width = self.data.shape

        dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64)
        dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512 // dimX)

        dimGridX = (height // dimX, depth // dimY)
        dimGridY = (width // dimX, depth // dimY)
        dimGridZ = (width // dimX, height // dimY)

        SamplesToCoefficients3DX(gca.gpudata, np.uint32(gca.strides[1]),
                                 np.uint32(width), np.uint32(height), np.uint32(depth),
                                 grid=dimGridX, block=(dimX, dimY, 1))

        SamplesToCoefficients3DY(gca.gpudata, np.uint32(gca.strides[1]),
                                 np.uint32(width), np.uint32(height), np.uint32(depth),
                                 grid=dimGridY, block=(dimX, dimY, 1))

        SamplesToCoefficients3DZ(gca.gpudata, np.uint32(gca.strides[1]),
                                 np.uint32(width), np.uint32(height), np.uint32(depth),
                                 grid=dimGridZ, block=(dimX, dimY, 1))

        cuda.Context.synchronize()

        # Rotation will be done many times, so it's better to make prepared calls
        self.cuda_transform = self.kernels.get_function('transform')
        self.cuda_transform.prepare('PPP')

        # CUDA kernel work parameters
        dim_blocks = (8, 8, 8)
        gridz = self.data.shape[2] // 8 + 1 * (self.data.shape[2] % 8 != 0)
        gridy = self.data.shape[1] // 8 + 1 * (self.data.shape[1] % 8 != 0)
        gridx = self.data.shape[0] // 8 + 1 * (self.data.shape[0] % 8 != 0)
        dim_grid = (gridz, gridy, gridx)  # in CUDA X is the fastest changing dimension, switching dimensions
        self.cuda_transform.dim_blocks = dim_blocks
        self.cuda_transform.dim_grid = dim_grid
        self.cuda_transform.dims = ga.to_gpu(np.array(self.data.shape[::-1], dtype=np.int32))
        self.cuda_coeff_tex_ref = self.kernels.get_texref('coeff_tex')

        # write gca to array
        with cuda_activate(self.coeff_tex_cuda) as ary:
            cpy = cuda.Memcpy3D()
            cpy.set_src_device(gca.gpudata)
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = gca.strides[1]
            cpy.src_height = cpy.height = height
            cpy.depth = depth
            cpy()

            # Bind cuda texture ref to opengl 3d prefilted texture
            self.cuda_coeff_tex_ref.set_array(ary)

        # free up temp data
        gca.gpudata.free()

    def cuda_rotate(self, m_transform: np.ndarray, profile=True):

        # make context active
        self.makeCurrent()

        # clear working volume: we are going to change it with CUDA
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.volume_buf)
        glClearBufferData(GL_PIXEL_UNPACK_BUFFER, GL_R32F, GL_RED, GL_FLOAT, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        time_start = timeit.default_timer()

        # map working volume buffer and get pointer to it
        volume_buf_map_obj = self.volume_buf_cuda.map()
        buf_data, buf_siz = volume_buf_map_obj.device_ptr_and_size()

        # run the transform kernel that fills PBO with transformed volume
        m_transform = ga.to_gpu(m_transform)
        self.cuda_transform.prepared_call(self.cuda_transform.dim_grid, self.cuda_transform.dim_blocks,
                                          self.cuda_transform.dims.gpudata,
                                          m_transform.gpudata,
                                          np.intp(buf_data))

        if profile:
            cuda.Context.synchronize()

        # unmap volume buffer
        volume_buf_map_obj.unmap()

        # CUDA profiling
        time_end = timeit.default_timer() - time_start
        if profile:
            print('Transform CUDA code finished in {:.4f} ms'.format(time_end * 1000))

        # # debug code to save stuff from working buffer
        # glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.volume_buf)
        # raw_data = glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, self.data.nbytes)
        # glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        # notraw_data = np.fromstring(raw_data, dtype=np.float32).reshape(self.data.shape)
        # import mrcfile as mrc
        # t = mrc.new(r'C:\Projects\ml\experiments\volume-viewer\test_buffer.mrc', notraw_data, overwrite=True)
        # t.close()

        # CUDA transformed populated PBO, now transfer from PBO to Texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.volume_buf)
        glBindTexture(GL_TEXTURE_2D, self.render_tex)

        # offset
        offset = ctypes.c_void_p(self.data.strides[0] * self.slice)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.data.shape[2], self.data.shape[1],
                        GL_RED, GL_FLOAT, offset)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # # debug code to save stuff from render texture
        # glBindTexture(GL_TEXTURE_2D, self.render_tex)
        # raw_data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
        # glBindTexture(GL_TEXTURE_2D, 0)
        # notraw_data = np.fromstring(raw_data, dtype=np.float32).reshape((self.data.shape[1], self.data.shape[2]))
        # import mrcfile as mrc
        # t = mrc.new(r'C:\Projects\ml\experiments\volume-viewer\test_texture.mrc', notraw_data, overwrite=True)
        # t.close()

        # re-draw
        self.update()

    def save_volume(self, path):
        print('Saving...\n')
        self.makeCurrent()

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.volume_buf)
        raw_data = glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, self.data.nbytes)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        data_formatted = np.fromstring(raw_data, dtype=np.float32).reshape(self.data.shape)
        t = mrc.new(path, data_formatted, overwrite=True)
        t.close()

        self.doneCurrent()
        print('Saved at', path, '\n')

    def change_slice(self, slice):
        self.slice = slice
        self.norm_slice = slice / self.data.shape[0]

        self.makeCurrent()

        # Transfer from PBO to Render Tex
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.volume_buf)
        glBindTexture(GL_TEXTURE_2D, self.render_tex)
        # offset in linear memory to the slice we want
        offset = ctypes.c_void_p(self.data.strides[0] * self.slice)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.data.shape[2], self.data.shape[1],
                        GL_RED, GL_FLOAT, offset)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Colormap
        slice_low, slice_high = np.percentile(self.data[self.slice], (0.5, 99.5))
        glUniform1f(self.prog.uniforms['u_min_i'], slice_low)
        glUniform1f(self.prog.uniforms['u_max_i'], slice_high)
        self.doneCurrent()

        self.update()

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()

def open_viewer(data):
    app = QApplication(sys.argv)
    app.setApplicationName('Volume Viewer')

    if isinstance(data, str):

        with warnings.catch_warnings():
            particle = mrc.open(data, permissive=True)
            volume = particle.data.astype(np.float32)
            particle.close()

        win = VolumeViewerWindow(volume * 10000)
        win.show()

        # run the loop
        sys.exit(app.exec_())
