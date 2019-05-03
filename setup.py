import distutils.log
import pathlib
import subprocess

import setuptools
import setuptools.command.build_py


class BuildPyCommand(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('qt_uic')
        setuptools.command.build_py.build_py.run(self)


class QtUicCommand(setuptools.Command):
    description = 'run pyside2-uic command to convert ui to py'
    user_options = []
    ui_paths = [
        pathlib.Path('kwiiyatta/view/qt/ui')
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        command = 'pyside2-uic'
        for p in self.ui_paths:
            for f in p.glob('*.ui'):
                commands = [
                    command,
                    str(f),
                    '-o', str(f.with_suffix('.py'))
                ]
                self.announce(
                    'Running command: %s' % str(commands),
                    level=distutils.log.INFO)
                subprocess.check_call(commands)


setuptools.setup(
    name="kwiiyatta",
    version="0.0.1",
    author="Iselix",
    description="Voice conversion tool",
    packages=setuptools.find_packages(),
    package_data={'kwiiyatta.view': ['res/*.ico']},
    install_requires=[
        'nnmnkwii',
        'numpy',
        'scipy',
        'sklearn',
        'pyaudio',
        'pyworld',
        'pysptk',
        'pyside2',
    ],
    entry_points={
        'console_scripts': [
            'kwiiyatta=kwiiyatta.convert_voice:main',
            'kwiieiya=kwiiyatta.resynthesize_voice:main',
        ],
    },
    cmdclass={
        'qt_uic': QtUicCommand,
        'build_py': BuildPyCommand,
    },
)
