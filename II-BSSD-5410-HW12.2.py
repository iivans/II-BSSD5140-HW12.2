from PIL import Image, ImageOps
import math

def apply_edge_detection_kernels(img, kernel_size=3, step=1):
    start = kernel_size // 2
    width, height = img.size
    con_pixels = []

    # Determine the output dimensions 
    output_width = (width - start * 2) // step
    output_height = (height - start * 2) // step
    dims = (output_width, output_height)

    for x in range(start, height - start, step):
        for y in range(start, width - start, step):
            # Get the 3x3 neighborhood pixels
            tl = img.getpixel((y - 1, x - 1))
            tc = img.getpixel((y, x - 1))
            tr = img.getpixel((y + 1, x - 1))
            lc = img.getpixel((y - 1, x))
            cc = img.getpixel((y, x))
            rc = img.getpixel((y + 1, x))
            bl = img.getpixel((y - 1, x + 1))
            bc = img.getpixel((y, x + 1))
            br = img.getpixel((y + 1, x + 1))

            # Bottom edge 
            sum_bottom = abs(tl * 1 + tc * 0 + tr * -1 +
                             lc * 2 + cc * 0 + rc * -2 +
                             bl * 1 + bc * 0 + br * -1)
            
            # Top edge 
            sum_top = abs(tl * -1 + tc * 0 + tr * 1 +
                          lc * -2 + cc * 0 + rc * 2 +
                          bl * -1 + bc * 0 + br * 1)
            
            # Left edge 
            sum_left = abs(tl * 1 + tc * 2 + tr * 1 +
                           lc * 0 + cc * 0 + rc * 0 +
                           bl * -1 + bc * -2 + br * -1)
            
            # Right edge 
            sum_right = abs(tl * -1 + tc * -2 + tr * -1 +
                            lc * 0 + cc * 0 + rc * 0 +
                            bl * 1 + bc * 2 + br * 1)

            final_ave = 0
            # Add non-zero averages to the final result
            if sum_bottom > 0:
                final_ave += sum_bottom
            if sum_top > 0:
                final_ave += sum_top
            if sum_left > 0:
                final_ave += sum_left
            if sum_right > 0:
                final_ave += sum_right

            # Append the final value to the convolution pixel list
            con_pixels.append(final_ave)

    # Create the output image with the correct dimensions
    output = Image.new('L', dims)
    output.putdata(con_pixels)
    output_resized = output.resize(img.size, Image.BILINEAR)
    output_resized.show()
    output_resized.save("convolved_output_edges.png")
    print("Output saved")

def main():
    og_image = Image.open("demo.png")
    gray_image = ImageOps.grayscale(og_image)
    gray_image.show()
    
    # Apply edge detection with kernel size 3 and step 1
    apply_edge_detection_kernels(gray_image, kernel_size=3, step=1)

if __name__ == "__main__":
    main()
