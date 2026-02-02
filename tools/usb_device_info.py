# -*- coding: utf-8 -*-
"""
USB Device Descriptor Viewer

Displays USB device descriptors including bInterval (polling interval)
for HID devices like USB-IO2.0.

Requirements:
    pip install pyusb

On Windows, you may need to install libusb:
    - Download zadig (https://zadig.akeo.ie/)
    - Or use libusb-win32

Usage:
    python usb_device_info.py
    python usb_device_info.py --vid 0x1352 --pid 0x0121
"""

import argparse
import sys

# USB-IO2.0 default IDs
DEFAULT_VID = 0x1352
DEFAULT_PID = 0x0121


def show_device_info_pyusb(vid: int, pid: int):
    """Show device info using pyusb (detailed)."""
    try:
        import usb.core
        import usb.util
    except ImportError:
        print("pyusb not installed. Install with: pip install pyusb")
        return False

    print(f"\n{'='*60}")
    print(f" USB Device Descriptor (pyusb)")
    print(f" VID: 0x{vid:04X}, PID: 0x{pid:04X}")
    print(f"{'='*60}\n")

    device = usb.core.find(idVendor=vid, idProduct=pid)

    if device is None:
        print(f"Device not found (VID=0x{vid:04X}, PID=0x{pid:04X})")
        return False

    print(f"Device Descriptor:")
    print(f"  bLength:            {device.bLength}")
    print(f"  bDescriptorType:    {device.bDescriptorType}")
    print(f"  bcdUSB:             {device.bcdUSB:#06x}")
    print(f"  bDeviceClass:       {device.bDeviceClass}")
    print(f"  bDeviceSubClass:    {device.bDeviceSubClass}")
    print(f"  bDeviceProtocol:    {device.bDeviceProtocol}")
    print(f"  bMaxPacketSize0:    {device.bMaxPacketSize0}")
    print(f"  idVendor:           0x{device.idVendor:04X}")
    print(f"  idProduct:          0x{device.idProduct:04X}")
    print(f"  bcdDevice:          {device.bcdDevice:#06x}")
    print(f"  iManufacturer:      {device.iManufacturer}")
    print(f"  iProduct:           {device.iProduct}")
    print(f"  iSerialNumber:      {device.iSerialNumber}")
    print(f"  bNumConfigurations: {device.bNumConfigurations}")

    try:
        print(f"\n  Manufacturer: {usb.util.get_string(device, device.iManufacturer)}")
        print(f"  Product:      {usb.util.get_string(device, device.iProduct)}")
    except Exception:
        pass

    # USB Speed
    speed_map = {
        0: "Unknown",
        1: "Low Speed (1.5 Mbps)",
        2: "Full Speed (12 Mbps)",
        3: "High Speed (480 Mbps)",
        4: "Super Speed (5 Gbps)",
    }
    try:
        speed = device.speed
        print(f"\n  USB Speed: {speed_map.get(speed, f'Unknown ({speed})')}")
    except Exception:
        pass

    # Configurations and Interfaces
    for cfg in device:
        print(f"\n  Configuration {cfg.bConfigurationValue}:")
        print(f"    bNumInterfaces:      {cfg.bNumInterfaces}")
        print(f"    bConfigurationValue: {cfg.bConfigurationValue}")
        print(f"    bmAttributes:        0x{cfg.bmAttributes:02X}")
        print(f"    bMaxPower:           {cfg.bMaxPower * 2} mA")

        for intf in cfg:
            print(f"\n    Interface {intf.bInterfaceNumber}, Alt {intf.bAlternateSetting}:")
            print(f"      bInterfaceClass:    {intf.bInterfaceClass} (HID={3})")
            print(f"      bInterfaceSubClass: {intf.bInterfaceSubClass}")
            print(f"      bInterfaceProtocol: {intf.bInterfaceProtocol}")
            print(f"      bNumEndpoints:      {intf.bNumEndpoints}")

            for ep in intf:
                ep_dir = "IN" if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_IN else "OUT"
                ep_type_map = {
                    usb.util.ENDPOINT_TYPE_CTRL: "Control",
                    usb.util.ENDPOINT_TYPE_ISO: "Isochronous",
                    usb.util.ENDPOINT_TYPE_BULK: "Bulk",
                    usb.util.ENDPOINT_TYPE_INTR: "Interrupt",
                }
                ep_type = ep_type_map.get(usb.util.endpoint_type(ep.bmAttributes), "Unknown")

                print(f"\n      Endpoint 0x{ep.bEndpointAddress:02X} ({ep_dir}):")
                print(f"        bmAttributes:     0x{ep.bmAttributes:02X} ({ep_type})")
                print(f"        wMaxPacketSize:   {ep.wMaxPacketSize}")
                print(f"        bInterval:        {ep.bInterval} ms")
                print(f"        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print(f"        This is the polling interval!")

    return True


def show_device_info_hid(vid: int, pid: int):
    """Show basic device info using hidapi."""
    try:
        import hid
    except ImportError:
        print("hidapi not installed. Install with: pip install hidapi")
        return False

    print(f"\n{'='*60}")
    print(f" USB HID Device Info (hidapi)")
    print(f" VID: 0x{vid:04X}, PID: 0x{pid:04X}")
    print(f"{'='*60}\n")

    # List all HID devices
    print("All HID devices:")
    for d in hid.enumerate():
        print(f"  VID=0x{d['vendor_id']:04X} PID=0x{d['product_id']:04X} "
              f"- {d['product_string']} ({d['manufacturer_string']})")

    print(f"\n{'-'*60}")
    print(f"Target device (VID=0x{vid:04X}, PID=0x{pid:04X}):\n")

    try:
        device = hid.device()
        device.open(vid, pid)

        print(f"  Manufacturer: {device.get_manufacturer_string()}")
        print(f"  Product:      {device.get_product_string()}")
        print(f"  Serial:       {device.get_serial_number_string()}")

        device.close()
        print("\n  Note: hidapi cannot show bInterval. Use pyusb for detailed info.")
        return True

    except Exception as e:
        print(f"  Failed to open device: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='USB Device Descriptor Viewer')
    parser.add_argument('--vid', type=lambda x: int(x, 0), default=DEFAULT_VID,
                        help=f'Vendor ID (default: 0x{DEFAULT_VID:04X})')
    parser.add_argument('--pid', type=lambda x: int(x, 0), default=DEFAULT_PID,
                        help=f'Product ID (default: 0x{DEFAULT_PID:04X})')
    parser.add_argument('--hid-only', action='store_true',
                        help='Only use hidapi (no pyusb)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" USB Device Descriptor Viewer")
    print("="*60)

    # Try hidapi first (usually works without driver changes)
    show_device_info_hid(args.vid, args.pid)

    # Try pyusb for detailed info
    if not args.hid_only:
        print("\n" + "-"*60)
        print("Attempting to get detailed info with pyusb...")
        print("(This may fail if libusb driver is not installed)")
        print("-"*60)

        if not show_device_info_pyusb(args.vid, args.pid):
            print("\nTo get detailed descriptor info (including bInterval):")
            print("  1. Install pyusb: pip install pyusb")
            print("  2. Install libusb driver using Zadig (https://zadig.akeo.ie/)")
            print("     - Run Zadig as Administrator")
            print("     - Options -> List All Devices")
            print("     - Select USB-IO2.0")
            print("     - Install libusb-win32 or WinUSB driver")
            print("  Note: This may break hidapi access until you restore the HID driver")


if __name__ == '__main__':
    main()
