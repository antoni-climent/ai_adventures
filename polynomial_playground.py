import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import sympy as sp

class PolynomialViewer:
    """
    This class is a polynomial viewer that allows you to plot a polynomial and 
    change the coefficients of the polynomial using sliders.
    """
    def __init__(self):
        # Create figure and axis
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Interactive Polynomial Plotter")
        
        # Initial expression
        self.expr_str = "a*x**2 + b*x + c"
        self.x_sym = sp.Symbol('x')
        self.params = {}
        
        # References to widgets to avoid garbage collection
        self.sliders = {}
        self.slider_axes = []
        self.text_box = None
        
        self.setup_ui()

    def setup_ui(self):
        # Clear figure for redraw 
        self.fig.clear()
        
        # Parse expression string
        try:
            self.expr = sp.sympify(self.expr_str)
        except Exception as e:
            self.expr = sp.sympify("0")
            print(f"Error parsing expression: {e}")
        
        # Find parameters (any symbol that is not 'x')
        free_symbols = [s for s in self.expr.free_symbols if s.name != 'x']
        free_symbols = sorted(free_symbols, key=lambda s: s.name)
        
        # Set default values for new parameters
        self.params = {s: self.params.get(s, 1.0) for s in free_symbols}
        
        # Calculate dynamic bottom margin depending on number of parameters
        n_params = len(self.params)
        bottom_margin = min(0.15 + 0.06 * n_params, 0.7)
        
        # Add main plot axis
        self.ax = self.fig.add_axes([0.1, bottom_margin, 0.8, 0.85 - bottom_margin])
        
        # Add textbox for polynomial input
        axbox = self.fig.add_axes([0.15, 0.90, 0.75, 0.05])
        self.text_box = TextBox(axbox, 'P(x) = ', initial=self.expr_str)
        self.text_box.on_submit(self.submit_expr)
        
        # Setup slider axes
        self.sliders = {}
        self.slider_axes = []
        
        slider_height = 0.03
        slider_spacing = 0.05
        
        for i, sym in enumerate(free_symbols):
            y_pos = bottom_margin - 0.08 - i * slider_spacing
            if y_pos < 0.02:
                print("Too many variables to fit on screen!")
                break
                
            ax_slider = self.fig.add_axes([0.15, y_pos, 0.7, slider_height])
            self.slider_axes.append(ax_slider)
            
            # Create slider
            slider = Slider(
                ax_slider, sym.name, -100.0, 100.0, valinit=self.params[sym], valstep=0.1
            )
            # Use default argument 's=sym' to bind the current symbol to the lambda
            slider.on_changed(lambda val, s=sym: self.update_param(s, val))
            self.sliders[sym] = slider
            
        self.draw_plot()
        
    def submit_expr(self, text):
        self.expr_str = text
        self.setup_ui()
        plt.draw()
        
    def update_param(self, sym, val):
        self.params[sym] = val
        self.draw_plot()
        
    def draw_plot(self):
        self.ax.clear()
        
        # Substitute the parameter values
        expr_subs = self.expr.subs(self.params)
        
        # Check if there are missing parameters
        remaining_syms = set(expr_subs.free_symbols) - {self.x_sym}
        if remaining_syms:
            self.ax.text(0.5, 0.5, f"Missing parameters: {remaining_syms}", ha='center')
            self.fig.canvas.draw_idle()
            return
            
        # Create a fast numerical function
        f = sp.lambdify(self.x_sym, expr_subs, modules=['numpy'])
        
        # Generate x values
        x_vals = np.linspace(-10, 10, 400)
        
        try:
            # Generate y values
            y_vals = f(x_vals)
            
            # Handle constant function case
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)
                
            # Plot
            self.ax.plot(x_vals, y_vals, 'b-', linewidth=2)
            self.ax.set_title(f"Graph of P(x) = {self.expr_str}")
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_xlim([-10, 10])
            
            # Add x and y axes lines
            self.ax.axhline(0, color='black', linewidth=1)
            self.ax.axvline(0, color='black', linewidth=1)
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error evaluating expression: {e}", ha='center')
            
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print("Starting Interactive Polynomial Plotter...")
    print("You can type an expression like 'a*x**2 + b*x + c' in the top text box.")
    print("Variables other than 'x' will automatically get their own sliders.")
    viewer = PolynomialViewer()
    plt.show()
