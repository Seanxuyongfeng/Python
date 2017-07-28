ifneq ($(ZEUSIS_TARGET_CUST),)
LOCAL_PREBUILD_CUST_DIR := device/zeusis/$(TARGET_PRODUCT)/cust/$(ZEUSIS_TARGET_CUST)
LOCAL_CUST_LINK_TARGET_APPS_DIR := /cust/common
LOCAL_TARGET_OUT_CUST := out/target/product/$(TARGET_PRODUCT)/cust
CUST_PYTHON_PATH := device/zeusis/$(TARGET_PRODUCT)/cust/main.py

# add common
$(shell mkdir -p $(LOCAL_TARGET_OUT_CUST)/common)
$(shell cp -rf $(LOCAL_PREBUILD_CUST_DIR)/common/* $(LOCAL_TARGET_OUT_CUST)/common)

$(foreach p,$(shell ls $(LOCAL_PREBUILD_CUST_DIR) | grep -E -v "^(common)"),\
    $(shell mkdir -p $(LOCAL_TARGET_OUT_CUST)/$(p); \
            cp -rf $(LOCAL_PREBUILD_CUST_DIR)/$(p)/* $(LOCAL_TARGET_OUT_CUST)/$(p))\
    $(foreach item, $(shell ls $(LOCAL_PREBUILD_CUST_DIR)/$(p)), \
            $(shell ln -sf $(LOCAL_CUST_LINK_TARGET_APPS_DIR) $(LOCAL_TARGET_OUT_CUST)/$(p)/$(item)/common; \
            python $(CUST_PYTHON_PATH) $(LOCAL_TARGET_OUT_CUST)/$(p)/$(item) "$(BUILD_ID)" \
            "$(BUILD_DISPLAY_ID)" "$(BUILD_NUMBER)" "$(BUILD_FINGERPRINT)" "$(build_desc)")))

endif