import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class CelestialBody:
    def __init__(self, position, velocity, mass, radius, color, name):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.name = name
        self.orbit = [self.position.copy()]
        
    def update_position(self, dt):
        self.position += self.velocity * dt
        self.orbit.append(self.position.copy())
        
    def update_velocity(self, acceleration, dt):
        self.velocity += acceleration * dt

class SolarSystem:
    def __init__(self):
        self.G = 1.0  # 중력 상수를 1.0으로 변경
        self.epsilon = 0.01  # 소프트닝 파라미터 감소
        self.bodies = []
        
    def add_body(self, body):
        self.bodies.append(body)
        
    def calculate_acceleration(self, body):
        acceleration = np.zeros(3)
        for other in self.bodies:
            if other != body:
                r = other.position - body.position
                distance = np.linalg.norm(r) + self.epsilon
                acceleration += self.G * other.mass * r / (distance**3)
        return acceleration
    
    def simulate_step(self, dt):
        accelerations = [self.calculate_acceleration(body) for body in self.bodies]
        for body, acc in zip(self.bodies, accelerations):
            body.update_velocity(acc, dt)
        for body in self.bodies:
            body.update_position(dt)

# 태양계 초기화
solar_system = SolarSystem()

# 초기 조건 수정 (blbadger/threebody 참고)
sun1 = CelestialBody(
    position=[0.97000436, -0.24308753, 0],
    velocity=[0.466203685, 0.43236573, 0],
    mass=1,
    radius=0.1,
    color='red',
    name='Sun 1'
)

sun2 = CelestialBody(
    position=[-0.97000436, 0.24308753, 0],
    velocity=[0.466203685, 0.43236573, 0],
    mass=1,
    radius=0.1,
    color='orange',
    name='Sun 2'
)

sun3 = CelestialBody(
    position=[0, 0, 0],
    velocity=[-0.93240737, -0.86473146, 0],
    mass=1,
    radius=0.1,
    color='yellow',
    name='Sun 3'
)

# 천체들을 태양계에 추가
for body in [sun1, sun2, sun3]:
    solar_system.add_body(body)

# 시각화 설정
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 플롯 범위 설정
max_range = 2  # 범위 축소
ax.set_xlim((-max_range, max_range))
ax.set_ylim((-max_range, max_range))
ax.set_zlim((-max_range, max_range))

# 애니메이션 업데이트 함수
def update(frame):
    # 시뮬레이션 스텝 실행
    dt = 0.001
    for _ in range(5):
        solar_system.simulate_step(dt)
    
    ax.cla()
    ax.set_xlim((-max_range, max_range))
    ax.set_ylim((-max_range, max_range))
    ax.set_zlim((-max_range, max_range))
    
    # 각 천체의 위치와 전체 궤적 업데이트
    for body in solar_system.bodies:
        orbit = np.array(body.orbit)  # 전체 궤적 사용
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 
                color=body.color, alpha=0.3, linewidth=0.5)
        ax.scatter(body.position[0], body.position[1], body.position[2],
                  color=body.color, s=50, label=body.name)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Three Body Problem (t={frame*dt*5:.3f})')
    ax.legend()
    
    # 고정된 시점 설정
    ax.view_init(elev=20, azim=45)

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=4000, interval=1, blit=False)

plt.show()
