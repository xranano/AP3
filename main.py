import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#function 1 and its derivative
def f1(x):
    return np.exp(-x) * np.sin(3 * x)
def f1_exact_derivative(x):
    return np.exp(-x) * (3 * np.cos(3 * x) - np.sin(3 * x))

#function 2 and its derivative
def f2(x, y):
    return x**2 * np.exp(-x**2 - y**2)
def f2_exact_partial_x(x, y):
    return 2*x * np.exp(-x**2 - y**2) * (1 - x**2)
def f2_exact_partial_y(x, y):
    return -2*x**2*y * np.exp(-x**2 - y**2)


#finate difference methods
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h
def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def forward_difference_2d(f, x0, y0, h, direction='x'):
    if direction == 'x':
        return (f(x0 + h, y0) - f(x0, y0)) / h
    else:
        return (f(x0, y0 + h) - f(x0, y0)) / h

def backward_difference_2d(f, x0, y0, h, direction='x'):
    if direction == 'x':
        return (f(x0, y0) - f(x0 - h, y0)) / h
    else:
        return (f(x0, y0) - f(x0, y0 - h)) / h

def central_difference_2d(f, x0, y0, h, direction='x'):
    if direction == 'x':
        return (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
    else:
        return (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)


#tangent line
def compute_tangent_line(x0, f, df, h_values, method='exact'):
    y0 = f(x0)

    if method == 'exact':
        slope = df(x0)
    elif method == 'forward':
        slope = forward_difference(f, x0, h_values)
    elif method == 'backward':
        slope = backward_difference(f, x0, h_values)
    elif method == 'central':
        slope = central_difference(f, x0, h_values)

    # Tangent line: y - y0 = m(x - x0)
    tangent = lambda x: y0 + slope * (x - x0)
    normal = np.array([-slope, 1])
    normal = normal / np.linalg.norm(normal)
    return slope, tangent, normal
def compute_tangent_plane_and_normal(x0, y0, f, dfx, dfy, h, method='exact'):
    z0 = f(x0, y0)

    if method == 'exact':
        fx = dfx(x0, y0)
        fy = dfy(x0, y0)
    elif method == 'forward':
        fx = forward_difference_2d(f, x0, y0, h, 'x')
        fy = forward_difference_2d(f, x0, y0, h, 'y')
    elif method == 'backward':
        fx = backward_difference_2d(f, x0, y0, h, 'x')
        fy = backward_difference_2d(f, x0, y0, h, 'y')
    elif method == 'central':
        fx = central_difference_2d(f, x0, y0, h, 'x')
        fy = central_difference_2d(f, x0, y0, h, 'y')

    # Normal vector: n = <-fx, -fy, 1> (or <fx, fy, -1>)
    normal = np.array([fx, fy, -1])
    normal = normal / np.linalg.norm(normal)  # Normalize

    # Tangent plane: z - z0 = fx(x - x0) + fy(y - y0)
    tangent_plane = lambda x, y: z0 + fx * (x - x0) + fy * (y - y0)

    return fx, fy, normal, tangent_plane


#visualisation

def plot_1d_function_and_tangent():
    x = np.linspace(0, 2 * np.pi, 500)
    x0 = np.pi / 2
    h = 0.1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle("Tangent Line and Finite Difference Derivative Comparison",
                 fontsize=15, weight='bold', color='#2C3E50')

    # === Left: Function + Tangent lines ===
    ax = axes[0]
    ax.plot(x, f1(x), color="#2E86AB", linewidth=2.2, label="f(x) = e^{-x}sin(3x)")

    methods = ['exact', 'forward', 'backward', 'central']
    colors = ['#000000', '#E74C3C', '#E67E22', '#27AE60']
    styles = ['-', '--', '-.', ':']
    x_tangent = np.linspace(x0 - 1, x0 + 1, 100)

    slopes = []
    normals = []

    for method, color, style in zip(methods, colors, styles):
        slope, tangent, normal = compute_tangent_line(x0, f1, f1_exact_derivative, h, method)
        slopes.append(slope)
        ax.plot(x_tangent, tangent(x_tangent), color=color, linestyle=style, linewidth=2,
                label=f'{method.capitalize()} (m={slope:.4f})')

        # Compute normal vector [-m, 1] and normalize
        n = np.array([-slope, 1])
        n = n / np.linalg.norm(n)
        normals.append(n)

        # Draw the normal vector as an arrow
        arrow_scale = 0.5  # adjust length of normal vector for visibility
        ax.arrow(
            x0, f1(x0),
            n[0] * arrow_scale, n[1] * arrow_scale,
            head_width=0.05, head_length=0.1, fc=color, ec=color, linewidth=1.8
        )

    ax.plot(x0, f1(x0), 'ko', markersize=8, label=f'Point ({x0:.2f}, {f1(x0):.4f})')


    # === Right plot (derivatives) ===
    ax2 = axes[1]
    ax2.plot(x, f1_exact_derivative(x), 'k-', linewidth=2.3, label='Exact')
    fd_forward = forward_difference(f1, x, h)
    fd_backward = backward_difference(f1, x, h)
    fd_central = central_difference(f1, x, h)
    ax2.plot(x, fd_forward, color='#E74C3C', linestyle='--', linewidth=1.8, alpha=0.8, label='Forward')
    ax2.plot(x, fd_backward, color='#E67E22', linestyle='-.', linewidth=1.8, alpha=0.8, label='Backward')
    ax2.plot(x, fd_central, color='#27AE60', linestyle=':', linewidth=2, alpha=0.9, label='Central')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("f'(x)", fontsize=12)
    ax2.set_title(f"Derivative Comparison (h = {h})", fontsize=13, weight='semibold', color='#2C3E50')
    ax2.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax2.grid(alpha=0.3, linestyle='--')

    # === Table (on same plot) ===
    errors = [abs(s - f1_exact_derivative(x0)) for s in slopes]
    col_labels = ['Method', 'Slope', 'Abs Error', 'Normal Vector [nx, ny]']
    table_data = [[m.capitalize(), f"{s:.6f}", f"{e:.2e}", f"[{n[0]:.4f}, {n[1]:.4f}]"]
                  for m, s, e, n in zip(methods, slopes, errors, normals)]

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='bottom',
                     cellLoc='center', colColours=['#f7f7f7'] * 4, bbox=[0.0, -0.7, 1.0, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)

    # Header style
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white')

    # Row colors
    colors = ['#ffcccb', '#ffd7b5', '#e6d5ff', '#c3f0ca']
    for i, color in enumerate(colors, start=1):
        for j in range(4):
            if (i, j) in table.get_celld():
                table[(i, j)].set_facecolor(color)

    for (row, col), cell in table.get_celld().items():
        cell.set_height(0.3)

    ax.text(0.5, -0.85, f'Tangent Slopes, Errors, and Normals at x₀ = {x0:.2f}',
            ha='center', va='center', fontsize=11, weight='bold', transform=ax.transAxes)
    ax.text(0.5, -0.95, f'Step size h = {h}', ha='center', va='center', fontsize=10, style='italic',
            transform=ax.transAxes)

    # === Axis and legend ===
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f"Tangent Lines at x₀ = {x0:.2f} (h = {h})", fontsize=13, weight='semibold', color='#2C3E50')
    ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')


    plt.subplots_adjust(bottom=0.6, wspace=0.7)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    return fig

def plot_2d_function_and_tangent_plane():
    x0, y0 = 0.5, 0.5
    h = 0.01

    fig = plt.figure(figsize=(20, 10))

    # Create mesh for plotting
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f2(X, Y)
    z0 = f2(x0, y0)

    # Compute all tangent planes and normals
    fx_exact, fy_exact, normal_exact, tangent_exact = compute_tangent_plane_and_normal(
        x0, y0, f2, f2_exact_partial_x, f2_exact_partial_y, h, 'exact')

    fx_forward, fy_forward, normal_forward, tangent_forward = compute_tangent_plane_and_normal(
        x0, y0, f2, f2_exact_partial_x, f2_exact_partial_y, h, 'forward')

    fx_backward, fy_backward, normal_backward, tangent_backward = compute_tangent_plane_and_normal(
        x0, y0, f2, f2_exact_partial_x, f2_exact_partial_y, h, 'backward')

    fx_central, fy_central, normal_central, tangent_central = compute_tangent_plane_and_normal(
        x0, y0, f2, f2_exact_partial_x, f2_exact_partial_y, h, 'central')

    # Plot 1: All tangent planes together
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.5, edgecolor='none')

    # Plot all tangent planes with different colors
    Z_exact = tangent_exact(X, Y)
    Z_forward = tangent_forward(X, Y)
    Z_backward = tangent_backward(X, Y)
    Z_central = tangent_central(X, Y)

    ax1.plot_surface(X, Y, Z_exact, color='red', alpha=0.3, label='Exact')
    ax1.plot_surface(X, Y, Z_forward, color='orange', alpha=0.25, label='Forward')
    ax1.plot_surface(X, Y, Z_backward, color='purple', alpha=0.25, label='Backward')
    ax1.plot_surface(X, Y, Z_central, color='green', alpha=0.25, label='Central')

    # Plot all normal vectors
    arrow_scale = 0.4
    ax1.quiver(x0, y0, z0, normal_exact[0], normal_exact[1], normal_exact[2],
               length=arrow_scale, color='red', linewidth=2.5, arrow_length_ratio=0.3)
    ax1.quiver(x0 + 0.05, y0, z0, normal_forward[0], normal_forward[1], normal_forward[2],
               length=arrow_scale, color='orange', linewidth=2, arrow_length_ratio=0.3)
    ax1.quiver(x0 - 0.05, y0, z0, normal_backward[0], normal_backward[1], normal_backward[2],
               length=arrow_scale, color='purple', linewidth=2, arrow_length_ratio=0.3)
    ax1.quiver(x0, y0 + 0.05, z0, normal_central[0], normal_central[1], normal_central[2],
               length=arrow_scale, color='green', linewidth=2, arrow_length_ratio=0.3)

    ax1.scatter([x0], [y0], [z0], color='black', s=100, zorder=10)
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_zlabel('z', fontsize=10)
    ax1.set_title(f'All Methods Combined (h={h})', fontsize=11, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Exact'),
        Line2D([0], [0], color='orange', linewidth=3, label='Forward'),
        Line2D([0], [0], color='purple', linewidth=3, label='Backward'),
        Line2D([0], [0], color='green', linewidth=3, label='Central')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Plot 2: Exact tangent plane
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
    ax2.plot_surface(X, Y, Z_exact, color='red', alpha=0.4)
    ax2.quiver(x0, y0, z0, normal_exact[0], normal_exact[1], normal_exact[2],
               length=0.5, color='red', linewidth=3, arrow_length_ratio=0.3)
    ax2.scatter([x0], [y0], [z0], color='black', s=100)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title(f'Exact\nn=[{normal_exact[0]:.4f}, {normal_exact[1]:.4f}, {normal_exact[2]:.4f}]', fontsize=9)
    ax2.view_init(elev=20, azim=45)

    # Plot 3: Forward difference
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
    ax3.plot_surface(X, Y, Z_forward, color='orange', alpha=0.4)
    ax3.quiver(x0, y0, z0, normal_forward[0], normal_forward[1], normal_forward[2],
               length=0.5, color='orange', linewidth=3, arrow_length_ratio=0.3)
    ax3.scatter([x0], [y0], [z0], color='black', s=100)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title(f'Forward\nn=[{normal_forward[0]:.4f}, {normal_forward[1]:.4f}, {normal_forward[2]:.4f}]', fontsize=9)
    ax3.view_init(elev=20, azim=45)

    # Plot 4: Backward difference
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
    ax4.plot_surface(X, Y, Z_backward, color='purple', alpha=0.4)
    ax4.quiver(x0, y0, z0, normal_backward[0], normal_backward[1], normal_backward[2],
               length=0.5, color='purple', linewidth=3, arrow_length_ratio=0.3)
    ax4.scatter([x0], [y0], [z0], color='black', s=100)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_title(f'Backward\nn=[{normal_backward[0]:.4f}, {normal_backward[1]:.4f}, {normal_backward[2]:.4f}]',
                  fontsize=9)
    ax4.view_init(elev=20, azim=45)

    # Plot 5: Central difference
    ax5 = fig.add_subplot(235, projection='3d')
    ax5.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
    ax5.plot_surface(X, Y, Z_central, color='green', alpha=0.4)
    ax5.quiver(x0, y0, z0, normal_central[0], normal_central[1], normal_central[2],
               length=0.5, color='green', linewidth=3, arrow_length_ratio=0.3)
    ax5.scatter([x0], [y0], [z0], color='black', s=100)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_title(f'Central\nn=[{normal_central[0]:.4f}, {normal_central[1]:.4f}, {normal_central[2]:.4f}]', fontsize=9)
    ax5.view_init(elev=20, azim=45)

    # Plot 6: Comparison table
    ax6 = fig.add_subplot(236)
    ax6.axis('off')

    table_data = [
        ['Exact', f'{fx_exact:.6f}', f'{fy_exact:.6f}', '-', '-'],
        ['Forward', f'{fx_forward:.6f}', f'{fy_forward:.6f}',
         f'{abs(fx_forward - fx_exact):.2e}', f'{abs(fy_forward - fy_exact):.2e}'],
        ['Backward', f'{fx_backward:.6f}', f'{fy_backward:.6f}',
         f'{abs(fx_backward - fx_exact):.2e}', f'{abs(fy_backward - fy_exact):.2e}'],
        ['Central', f'{fx_central:.6f}', f'{fy_central:.6f}',
         f'{abs(fx_central - fx_exact):.2e}', f'{abs(fy_central - fy_exact):.2e}']
    ]

    table = ax6.table(cellText=table_data,
                      colLabels=['Method', '∂f/∂x', '∂f/∂y', 'Error ∂x', 'Error ∂y'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    colors = ['#ffcccb', '#ffd7b5', '#e6d5ff', '#c3f0ca']
    for i, color in enumerate(colors, start=1):
        for j in range(5):
            table[(i, j)].set_facecolor(color)

    ax6.text(0.5, 0.95, f'Partial Derivatives at ({x0}, {y0})',
             ha='center', va='top', fontsize=12, weight='bold',
             transform=ax6.transAxes)
    ax6.text(0.5, 0.05, f'Step size h = {h}',
             ha='center', va='bottom', fontsize=10, style='italic',
             transform=ax6.transAxes)

    plt.tight_layout()
    return fig




if __name__ == "__main__":
    fig1 = plot_1d_function_and_tangent()
    # plt.savefig('1d_tangent_lines.png', dpi=150, bbox_inches='tight')
    fig2 = plot_2d_function_and_tangent_plane()
    # plt.savefig('2d_tangent_planes.png', dpi=150, bbox_inches='tight')


    # 1D results
    x0 = np.pi / 2
    h = 0.01
    print(f"\n1D FUNCTION at x = {x0:.4f} (h = {h}):")
    print(f"  f(x) = {f1(x0):.6f}")
    print(f"  Exact derivative: {f1_exact_derivative(x0):.6f}")
    print(f"  Forward difference: {forward_difference(f1, x0, h):.6f}")
    print(f"  Backward difference: {backward_difference(f1, x0, h):.6f}")
    print(f"  Central difference: {central_difference(f1, x0, h):.6f}")

    # 2D results
    x0, y0 = 0.5, 0.5
    print(f"\n2D FUNCTION at (x, y) = ({x0}, {y0}) (h = {h}):")
    print(f"  f(x,y) = {f2(x0, y0):.6f}")

    fx_exact = f2_exact_partial_x(x0, y0)
    fy_exact = f2_exact_partial_y(x0, y0)
    print(f"\n  Exact ∂f/∂x: {fx_exact:.6f}")
    print(f"  Forward ∂f/∂x: {forward_difference_2d(f2, x0, y0, h, 'x'):.6f}")
    print(f"  Central ∂f/∂x: {central_difference_2d(f2, x0, y0, h, 'x'):.6f}")

    print(f"\n  Exact ∂f/∂y: {fy_exact:.6f}")
    print(f"  Forward ∂f/∂y: {forward_difference_2d(f2, x0, y0, h, 'y'):.6f}")
    print(f"  Central ∂f/∂y: {central_difference_2d(f2, x0, y0, h, 'y'):.6f}")

    # Normal vectors
    print("\n  NORMAL VECTORS:")
    for method in ['exact', 'forward', 'central']:
        _, _, normal, _ = compute_tangent_plane_and_normal(
            x0, y0, f2, f2_exact_partial_x, f2_exact_partial_y, h, method)
        print(f"    {method.capitalize():10s}: [{normal[0]:8.5f}, {normal[1]:8.5f}, {normal[2]:8.5f}]")

    plt.show()