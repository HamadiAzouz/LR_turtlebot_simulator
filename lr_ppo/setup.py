from setuptools import setup, find_packages

package_name = 'lr_ppo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'torch',
        'gymnasium',
        'matplotlib',
        'tensorboard',
    ],
    zip_safe=True,
    maintainer='ROSI Student',
    maintainer_email='student@example.com',
    description='PPO Reinforcement Learning for ROSI TurtleBot3 maze navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_rosi = scripts.train_rosi:main',
            'evaluate_rosi = scripts.evaluate_rosi:main',
            'test_environment = scripts.test_environment:main',
        ],
    },
)
